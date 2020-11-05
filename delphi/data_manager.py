import queue
import numpy as np
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Iterable, Callable, Optional, Tuple

from logzero import logger

from delphi.context.data_manager_context import DataManagerContext
from delphi.model_trainer import DataRequirement
from delphi.proto.internal_pb2 import ExampleMetadata, GetExamplesRequest, GetExampleRequest
from delphi.proto.delphi_pb2 import LabeledExample, ExampleSet, LabeledExampleRequest
from delphi.utils import get_example_key, to_iter

IGNORE_FILE = 'ignore'
TRAIN_TO_TEST_RATIO = 4  # Hold out 20% of labeled examples as test


class DataManager(object):

    def __init__(self, context: DataManagerContext):
        self._context = context

        self._examples_dir = self._context.data_dir
        for example_set in ExampleSet.keys():
            example_dir = self._examples_dir / example_set.lower()
            example_dir.mkdir(parents=True, exist_ok=True)

        self._examples_lock = threading.Lock()
        self._example_counts = defaultdict(int)

        self._stored_examples_event = threading.Event()
        threading.Thread(target=self._promote_examples, name='promote-examples').start()

    def add_labeled_examples(self, examples: Iterable[LabeledExample]) -> None:
        if self._context.node_index == 0:
            self._store_labeled_examples(examples, None)
            return

        example_queue = queue.Queue()

        data_requirement = self._get_data_requirement()
        if data_requirement is DataRequirement.MASTER_ONLY:
            future = self._context.nodes[0].api.AddLabeledExamples.future(to_iter(example_queue))
            example_queue.put(LabeledExampleRequest(searchId=self._context.search_id))
            for example in examples:
                example_queue.put(LabeledExampleRequest(example=example))
        else:
            assert False
            future = self._context.nodes[0].api.AddLabeledExamples.future(to_iter(example_queue))
            example_queue.put(LabeledExampleRequest(searchId=self._context.search_id))

            if data_requirement is DataRequirement.DISTRIBUTED_FULL:
                self._store_labeled_examples(examples, lambda x: example_queue.put(LabeledExampleRequest(example=x)))
            else:
                # We're using distributed_positives - only share positives and test set
                def add_example(example: LabeledExample) -> None:
                    if example.exampleSet.value is ExampleSet.TEST or example.label == '1':
                        example_queue.put(LabeledExampleRequest(example=example))

                self._store_labeled_examples(examples, add_example)

        example_queue.put(None)
        future.result()

    @contextmanager
    def get_examples(self, example_set: ExampleSet) -> Iterable[Path]:
        with self._examples_lock:
            if self._context.node_index != 0:
                if example_set is ExampleSet.LABELED:
                    assert self._get_data_requirement() is not DataRequirement.MASTER_ONLY

                self._sync_with_master(example_set)
                yield self._examples_dir / self._to_dir(example_set)
            else:
                example_dir = self._examples_dir / self._to_dir(example_set)
                yield example_dir

    def get_example_stream(self, example_set: ExampleSet, node_index: int) -> Iterable[ExampleMetadata]:
        assert self._examples_lock.locked()
        assert self._context.node_index == 0
        assert node_index != 0
        example_dir = self._examples_dir / self._to_dir(example_set)

        if example_set is ExampleSet.LABELED:
            assert self._get_data_requirement() is not DataRequirement.MASTER_ONLY
            for label in example_dir.iterdir():
                if self._get_data_requirement() is DataRequirement.DISTRIBUTED_POSITIVES and label.name != '1':
                    continue

                for example in label.iterdir():
                    yield ExampleMetadata(label=label.name, key=example.name)
        elif example_set is ExampleSet.TEST:
            for label in example_dir.iterdir():
                label_examples = list(label.iterdir())
                for i in range(node_index - 1, len(label_examples), len(self._context.nodes) - 1):
                    yield ExampleMetadata(label=label.name, key=label_examples[i].name)
        else:
            raise NotImplementedError('Unknown example set: ' + self._to_dir(example_set))

    def get_example_path(self, example_set: ExampleSet, label: str, example: str) -> Path:
        assert self._examples_lock.locked()
        assert self._context.node_index == 0
        return self._examples_dir / self._to_dir(example_set) / label / example

    def reset(self, train_only: bool):
        # with self._staging_lock:
        #     self._clear_dir(self._staging_dir, train_only)

        # with self._examples_lock:
        #     self._clear_dir(self._examples_dir, train_only)
        return

    def _clear_dir(self, dir_path: Path, train_only: bool):
        for child in dir_path.iterdir():
            if child.is_dir():
                if child.name != 'test' or not train_only:
                    self._clear_dir(child, train_only)
            else:
                child.unlink()

    def _store_labeled_examples(self, examples: Iterable[LabeledExample],
                                callback: Optional[Callable[[LabeledExample], None]]) -> None:
        with self._examples_lock:
            # old_dirs = []
            # for dir in self._staging_dir.iterdir():
            #     if dir.name != IGNORE_FILE:
            #         for label in dir.iterdir():
            #             old_dirs.append(label)
            logger.debug("Saving Labeled Examples")
            for example in examples:
                content = b''
                if example.HasField('content'):
                    example_file = get_example_key(example.content)
                    content = example.content
                elif example.HasField('path'):
                    example_file = example.path
                    assert os.path.exists(example_file)
                    with open(example_file, "rb") as f:
                        content = f.read()
                    # os.remove(example_file)
                    example_file = os.path.basename(example_file)
                else:
                    raise NotImplementedError('unknown field: {}'.format(json_format.MessageToJson(example)))
                # self._remove_old_paths(example_file, old_dirs)

                if example.label != '-1':
                    if example.HasField('exampleSet'):
                        example_subdir = self._to_dir(example.exampleSet.value)
                    else:
                        example_subdir = 'unspecified'

                    label_dir = self._examples_dir / example_subdir / example.label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    example_path = label_dir / example_file
                    with example_path.open('wb') as f:
                        f.write(content)
                    # logger.info('Saved example with label {} to path {}'.format(example.label, example_path))
                else:
                    assert False, "Example {}".format(example)
                    logger.info('Example set to ignore - skipping')
                    ignore_file = self._examples_dir / IGNORE_FILE
                    with ignore_file.open('a+') as f:
                        f.write(example_file + '\n')

                if callback is not None:
                    callback(example)

        self._stored_examples_event.set()

    def _sync_with_master(self, example_set: ExampleSet) -> None:
        """
            Sync Examples at nodes with Master: For Distributed Data Management
        """
        assert False, "Disable sync for cvat"
        to_delete = defaultdict(set)
        example_dir = self._examples_dir / self._to_dir(example_set)
        for label in example_dir.iterdir():
            if self._get_data_requirement() is DataRequirement.DISTRIBUTED_POSITIVES \
                    and example_set is ExampleSet.LABELED \
                    and label.name != '1':
                continue

            to_delete[label.name] = set(x.name for x in label.iterdir())

        for example in self._context.nodes[0].internal.GetExamples(
                GetExamplesRequest(searchId=self._context.search_id, exampleSet=example_set,
                                   nodeIndex=self._context.node_index)):
            if example.key in to_delete[example.label]:
                to_delete[example.label].remove(example.key)
            else:
                example_content = self._context.nodes[0].internal.GetExample(
                    GetExampleRequest(searchId=self._context.search_id, exampleSet=example_set, label=example.label,
                                      key=example.key))
                label_dir = example_dir / example.label
                label_dir.mkdir(parents=True, exist_ok=True)
                example_path = label_dir / example.key
                with example_path.open('wb') as f:
                    f.write(example_content.value)

        for label in to_delete:
            for file in to_delete[label]:
                example_path = example_dir / label / file
                example_path.unlink()

    def _promote_examples(self):
        while True:
            try:
                self._stored_examples_event.wait()
                self._stored_examples_event.clear()

                new_positives = 0
                new_negatives = 0
                with self._examples_lock:
                    set_dirs = {}
                    for example_set in [ExampleSet.LABELED, ExampleSet.TEST]:
                        example_dir = self._examples_dir / self._to_dir(example_set)
                        set_dirs[example_set] = list(example_dir.iterdir())

                    for file in self._examples_dir.iterdir():
                        if file.name == IGNORE_FILE:
                            assert False
                            with file.open() as ignore_file:
                                for line in ignore_file:
                                    for example_set in set_dirs:
                                        old_path = self._remove_old_paths(line, set_dirs[example_set])
                                        if old_path is not None:
                                            self._increment_example_count(example_set, old_path.parent.name, -1)
                        else:
                            dir_positives, dir_negatives = self._promote_examples_dir(file, set_dirs)
                            new_positives += dir_positives
                            new_negatives += dir_negatives
                self._context.new_examples_callback(new_positives, new_negatives)
            except Exception as e:
                logger.exception(e)

    def _promote_examples_dir(self, subdir: Path, set_dirs: Dict['ExampleSet', List[Path]]) -> Tuple[int, int]:

        if (subdir.name != self._to_dir(ExampleSet.LABELED) and
            subdir.name != self._to_dir(ExampleSet.TEST)):
            return (0, 0)

        new_positives = 0
        new_negatives = 0
        for label in subdir.iterdir():
            example_files = list(label.iterdir())
            if label.name == '1':
                new_positives += len(example_files)
            else:
                new_negatives += len(example_files)

            for example_file in example_files:
                for example_set in set_dirs:
                    # old_path = self._remove_old_paths(example_file.name, set_dirs[example_set])
                    path_exists = np.any([(old_dir/example_file.name).exists()
                                    for old_dir in set_dirs[example_set]])
                    if path_exists:
                        self._increment_example_count(example_set, label.name, -1)

                if subdir.name == 'test' or (subdir.name == 'unspecified'
                                             and self._get_example_count(ExampleSet.TEST,
                                                                         label.name) * TRAIN_TO_TEST_RATIO <
                                             self._get_example_count(ExampleSet.LABELED, label.name)):
                    example_set = ExampleSet.TEST
                else:
                    example_set = ExampleSet.LABELED

                self._increment_example_count(example_set, label.name, 1)
                # example_dir = self._examples_dir / self._to_dir(example_set) / label.name
                # example_dir.mkdir(parents=True, exist_ok=True)
                # example_path = example_dir / example_file.name
                # logger.info("Rename {} {}".format(example_file, example_path))
                # example_file.rename(example_path)
                # logger.info('Promoted example with label {} to path {}'.format(label.name, example_path))

        return new_positives, new_negatives


    def _get_data_requirement(self) -> DataRequirement:
        return max([x.data_requirement for x in self._context.get_active_trainers()], key=lambda y: y.value)

    def _get_example_count(self, example_set: ExampleSet, label: str) -> int:
        return self._example_counts['{}_{}'.format(ExampleSet.Name(example_set), label)]

    def _increment_example_count(self, example_set: ExampleSet, label: str, delta: int) -> None:
        self._example_counts['{}_{}'.format(ExampleSet.Name(example_set), label)] += delta

    @staticmethod
    def _remove_old_paths(example_file: str, old_dirs: List[Path]) -> Optional[Path]:
        logger.info(f"{example_file} {old_dirs}")
        assert False
        for old_path in old_dirs:
            old_example_path = old_path / example_file
            if old_example_path.exists():
                old_example_path.unlink()
                logger.info('Removed old path {} for example'.format(old_example_path))
                return old_example_path
        return None

    @staticmethod
    def _to_dir(example_set: ExampleSet):
        return ExampleSet.Name(example_set).lower()
