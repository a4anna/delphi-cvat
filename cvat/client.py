import grpc
import os
import signal
import sys
import yaml
import uuid
import time
import threading
import multiprocessing_logging
from pathlib import Path
from logzero import logger
from google.protobuf.empty_pb2 import Empty

from delphi.proto.delphi_pb2 import InferRequest, InferResult, ModelStats, DirectoryDataset,\
    ImportModelRequest, ModelArchive, LabeledExampleRequest, SearchId, \
    AddLabeledExampleIdsRequest, LabeledExample, DelphiObject, GetObjectsRequest, SearchStats, SearchInfo, \
    CreateSearchRequest, Dataset, ReexaminationStrategyConfig, NoReexaminationStrategyConfig

from delphi.proto.delphi_pb2 import ModelConditionConfig, ExamplesPerLabelConditionConfig, \
     ModelConfig, SVMConfig, SVMMode, PercentageThresholdPolicyConfig, ExampleSetWrapper,\
     SelectorConfig, RetrainPolicyConfig, TopKSelectorConfig, ThresholdConfig, ExampleSet

from delphi.proto.delphi_pb2_grpc import DelphiServiceStub
from delphi.utils import log_exceptions
from cvat.result_manager import ResultManager


class CvatClient(object):

    def __init__(self, config):
        multiprocessing_logging.install_mp_handler()
        self.config = config
        channel = grpc.insecure_channel('localhost:{}'.format(self.config['port']))
        self.stub  = DelphiServiceStub(channel)
        self.train_dir = os.path.join(self.config['root_dir'], "labeled")
        logger.debug("Labeled Directory {}".format(self.train_dir))
        self.search_id = SearchId(value=str(uuid.uuid4()))
        self.result_manager = ResultManager(self.stub, self.search_id, self.train_dir, config['cvat'])
        signal.signal(signal.SIGINT, self.stop)

    def create_search(self):
        nodes = ['localhost']
        train_strategy = [ModelConditionConfig(
                            examplesPerLabel=ExamplesPerLabelConditionConfig(
                                count=5,
                                model=ModelConfig(
                                    svm=SVMConfig(
                                        mode=SVMMode.MASTER_ONLY,
                                        featureExtractor="mobilenet_v2",
                                        probability=True,
                                        linearOnly=True,
                                    )
                                )
                            ))]
        retrain_policy = RetrainPolicyConfig(percentage=PercentageThresholdPolicyConfig(
                            threshold=0.1,
                            onlyPositives=False,
        ))
        dataset = Dataset(directory=DirectoryDataset(
                            name=os.path.join(self.config['root_dir'], "unlabeled"),
                            loop=self.config['data_loop']
        ))
        reex = ReexaminationStrategyConfig(none=NoReexaminationStrategyConfig(k=0))
        if "threshold" in self.config['selector']:
            param = self.config['selector']['threshold']
            selector = SelectorConfig(threshold=ThresholdConfig(threshold=param['threshold'],
                            reexaminationStrategy=reex))
        else:
            param = self.config['selector']['topk']
            selector = SelectorConfig(topk=TopKSelectorConfig(k=param['k'], batchSize=param['batchSize'],
                            reexaminationStrategy=reex))
        has_init_examples = True #os.path.exists(os.path.join(self.config['root_dir'], "data", "train"))

        search = CreateSearchRequest(
                    searchId=self.search_id,
                    nodes=nodes,
                    nodeIndex=0,
                    trainStrategy=train_strategy,
                    retrainPolicy=retrain_policy,
                    onlyUseBetterModels=False,
                    dataset=dataset,
                    selector=selector,
                    hasInitialExamples=True,
                    metadata="positive"
        )
        self.stub.CreateSearch(search)
        logger.debug("Create Delphi Search")
        self.stub.AddLabeledExamples(self.add_initial_examples())


    def add_initial_examples(self):
        yield LabeledExampleRequest(searchId=self.search_id)
        num_examples = min([len(list(Path(example_dir).glob('*')))
                        for example_dir in Path(self.train_dir).iterdir()])
        for example_dir in Path(self.train_dir).iterdir():
            label = example_dir.name
            for i, path in enumerate(example_dir.iterdir()):
                if i >= num_examples:
                    break
                example = LabeledExampleRequest(example=LabeledExample(
                    label=label,
                    exampleSet=ExampleSetWrapper(value=ExampleSet.LABELED),
                    path=str(path),
                ))
                yield example


    def start(self):
        self.create_search()
        self.stub.StartSearch(self.search_id)
        try:
            threading.Thread(target=self._result_thread, name='get-results').start()
        except Exception as e:
            self.stop()
            raise e

    def stop(self, *args):
        logger.info("Stop called")
        self.stub.StopSearch(self.search_id)
        time.sleep(5)
        self.result_manager.terminate()
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)


    def _result_thread(self):
        while True:
            if (self.result_manager.tasks_lock._value == 0):
                self.stub.PauseSearch(self.search_id)
                while(self.result_manager.tasks_lock._value == 0):
                    pass
                self.stub.RestartSearch(self.search_id)
            for result in self.stub.GetResults(self.search_id):
                self.result_manager.add(result.objectId)

            if not self.result_manager.running:
                self.stop()
