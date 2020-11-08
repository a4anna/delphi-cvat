import os
import threading
import logging
import requests
import sys
import shutil
import time
import numpy as np
from http.client import HTTPConnection
from threading import Timer
from logzero import logger

from delphi.proto.delphi_pb2 import LabeledExampleRequest, LabeledExample
from delphi.proto.delphi_pb2 import ExampleSetWrapper, ExampleSet

from cvat.cvat import CLI, CVAT_API_V1

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


class ResultManager(object):

    def __init__(self, stub, search_id, train_path, cvat_config):
        task_config = cvat_config['tasks']
        self.stub = stub
        self.search_id = search_id
        self.train_path = train_path
        self.results = []
        self.results_len = task_config['length']
        self.results_wait = task_config['time']

        self.pending_tasks = {}
        self.finished_ids = set()
        self.seen_results = set()
        self.total_tasks = max(int(task_config.get('pending', '3')), 1)
        self.max_length = self.total_tasks * self.results_len
        self.curr_task_id = 0

        self.results_lock = threading.Lock()
        self.train_lock = threading.Lock()
        self._tasks_lock = threading.Semaphore(self.total_tasks)

        self.running = True

        self.labels = [{'attributes': [], 'name': 'positive'}]
        with requests.Session() as session:
            self.api = CVAT_API_V1('%s:%s' % (cvat_config['host'], cvat_config['port']))
            self.cli = CLI(session, self.api, [cvat_config['user'], cvat_config['password']])

        self._status_monitor = RepeatedTimer(60, self._monitor_tasks)
        self._result_check = RepeatedTimer(2, self._check_results)
        self.time_start = time.time()
        self.terminate_counter = 0
        self.termination_limit = 3

    def on_running(func):
        def wrapper(self, *args, **kwargs):
            if self.running:
                return func(self, *args, **kwargs)
        return wrapper

    def is_full(self):
        if len(self.results) >= self.max_length:
            return True
        return False

    @on_running
    def add(self, result):
        with self.results_lock:
            while self.is_full():
                continue
            result_id = result[0]
            if os.path.basename(result_id) not in self.seen_results:
                self.results.append(result)

    @on_running
    def _check_length(self):
        with self.results_lock:
            if ((len(self.results) < self.results_len) and
                (round(time.time() - self.time_start) < self.results_wait)):
                return
            self._create_task()
        if self.terminate_counter >= self.termination_limit:
            self.running = False

    def all_task_locks_free(self):
        return self._tasks_lock._value == self.total_tasks

    def all_task_locks_acquired(self):
        return self._tasks_lock._value == 0

    @on_running
    def _create_task(self):
        if not self.results:
            time_lapsed = round(time.time() - self.time_start)
            if time_lapsed > self.results_wait:
                self.terminate_counter += 1
                logger.debug("Waiting to terminate {}/{}".format(
                              self.terminate_counter, self.termination_limit))
            return
        self._tasks_lock.acquire()
        task_results = self.results[:self.results_len]
        self.results = self.results[self.results_len:]
        try:
            result_ids, versions = zip(*task_results)
            result_ids = sorted(set(result_ids))
            model_version = min(versions)
            self.curr_task_id += 1
            task_name = f"task-{self.curr_task_id}-model-{model_version}"
            result_ids = [p for p in result_ids if os.path.exists(p)
                            and (os.path.basename(p) not in self.seen_results)]

            [self.seen_results.add(os.path.basename(p)) for p in result_ids]
            if not result_ids:
                self.terminate_counter += 1
                return
            with self.train_lock:
                task_id = self.cli.tasks_create(task_name, self.labels, result_ids)
                self.pending_tasks[task_id] = result_ids
            self.time_start = time.time()
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            self._tasks_lock.release()
            logger.critical(e)

    def _monitor_tasks(self):
        self._check_status()

    def _check_results(self):
        self._check_length()

    def _get_completed_tasks(self, task_ids):
        stats = self.cli.tasks_status(task_ids)
        completed_stats = filter(lambda x: x["status"] == "completed", stats)
        completed_ids = [s["id"] for s in completed_stats]
        completed_ids = list(set(completed_ids).difference(self.finished_ids))
        return completed_ids

    @on_running
    def _check_status(self):
        with self.train_lock:
            task_ids = self.pending_tasks.keys()
            if not task_ids:
                return
            try:
                for i in self._get_completed_tasks(task_ids):
                    task_id, image_ids = self.cli.tasks_dump(i)
                    self._add_train(task_id, image_ids)
                    # updating task start time to prevent a new task
                    # creation immediatelly after releasing lock
                    self.time_start = time.time()
                    self._tasks_lock.release()

            except (requests.exceptions.HTTPError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException) as e:
                logger.critical(e)

    def _add_train(self, task_id, positive_ids):
        if ((task_id not in self.pending_tasks) or
           (task_id in self.finished_ids)):
            return

        task_images = self.pending_tasks[task_id]
        del self.pending_tasks[task_id]

        all_results = np.array(task_images)
        positive_ids = np.asarray(positive_ids, dtype=np.int32)
        mask_ids = np.zeros(all_results.size, dtype=bool)
        mask_ids[positive_ids] = True
        positives = all_results[mask_ids]
        negatives = all_results[~mask_ids]

        labeled_examples = []
        for positive in positives:
            label = '1'
            dst_path = os.path.join(self.train_path, label, os.path.basename(positive))
            if os.path.exists(positive):
                shutil.move(positive, dst_path)
        for negative in negatives:
            label = '0'
            dst_path = os.path.join(self.train_path, label, os.path.basename(negative))
            if os.path.exists(negative):
                shutil.move(negative, dst_path)

        example = self._get_example(dst_path, label)
        if example:
            labeled_examples.append(example)

        [self.seen_results.remove(os.path.basename(p)) for p in all_results]
        self.stub.AddLabeledExamples(self._add_examples(labeled_examples))
        self.finished_ids.add(task_id)
        time.sleep(5)

    def _get_example(self, path, label):
        if not os.path.exists(path):
            return None
        return LabeledExampleRequest(example=LabeledExample(
                    label=label,
                    exampleSet=ExampleSetWrapper(value=ExampleSet.LABELED),
                    path=path,
               ))

    def _add_examples(self, examples):
        yield LabeledExampleRequest(searchId=self.search_id)
        for example in examples:
            yield example

    def terminate(self):
        self.running = False
        self._result_check.stop()
        self._status_monitor.stop()
        for _ in range(self.total_tasks):
            self._tasks_lock.release()
        # import psutil
        # current_process_pid = psutil.Process().pid
        import signal
        pid = os.getpid()
        os.kill(pid, signal.SIGKILL)
