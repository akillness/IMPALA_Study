import torch.multiprocessing as mp

import threading
import queue

class SyncParameters(object):
    def __init__(self, lock):
        self.lock = lock
        self.weight = None

    def pull(self):
        with self.lock:
            return self.weight

    def push(self, weigth):
        with self.lock:
            self.weight = weigth

class EnvThread(object):
    def __init__(self, env_class, constructor_kwargs):
        self.env_class = env_class
        self._constructor_kwargs = constructor_kwargs

    def start(self):
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._thread = threading.Thread(target=self.worker, args=(self.env_class, self._constructor_kwargs, self.command_queue, self.result_queue))
        self._thread.start()
        result = self.result_queue.get()
        if isinstance(result, Exception):
            raise result

    def close(self):
        try:
            self.command_queue.put([2, None])
            self._thread.join()
        except IOError:
            raise IOError
        print("closed env type of normal")

    def reset(self):
        self.command_queue.put([0, None])
        state = self.result_queue.get()
        if state is None:
            raise ValueError
        return state

    def step(self, action):
        self.command_queue.put([1, action])
        state, reward, terminal = self.result_queue.get()
        return state, reward, terminal

    def worker(self, env_class, constructor_kwargs, command_queue, result_queue):
        try:
            env = env_class(**constructor_kwargs)
            result_queue.put(None)  # Ready.
            while True:
                # Receive request.
                command, arg = command_queue.get()
                if command == 0:
                    result_queue.put(env.reset())
                    result_queue.task_done()
                elif command == 1:
                    result_queue.put(env.step(arg))
                    result_queue.task_done()
                elif command == 2:
                    env.close()
                    break
                else:
                    print("bad command: {}".format(command))
        except Exception as e:
            if 'env' in locals() and hasattr(env, 'close'):
                try:
                    env.close()
                    print("closed error")
                except:
                    pass
            result_queue.put(e)


class EnvProcess(object):
    def __init__(self, env_class, constructor_kwargs):
        self.env_class = env_class
        self._constructor_kwargs = constructor_kwargs

    def start(self):
        self.parent_conn, child_conn = mp.Pipe()
        self._process = mp.Process(target=self.worker, args=(self.env_class, self._constructor_kwargs, child_conn))
        self._process.start()
        result = self.parent_conn.recv()
        if isinstance(result, Exception):
            raise result

    def close(self):
        try:
            self.parent_conn.send((2, None))
            self.parent_conn.close()
        except IOError:
            raise IOError
        print("closed env type of normal")
        self._process.join()

    def reset(self):
        self.parent_conn.send([0, None])
        state = self.parent_conn.recv()
        if state is None:
            raise ValueError
        return state

    def step(self, action):
        self.parent_conn.send([1, action])
        state, reward, terminal = self.parent_conn.recv()
        return state, reward, terminal

    def worker(self, env_class, constructor_kwargs, conn):
        try:
            env = env_class(**constructor_kwargs)
            conn.send(None)
            while True:
                command, arg = conn.recv()
                if command == 0:
                    conn.send(env.reset())
                elif command == 1:
                    conn.send(env.step(arg))
                elif command == 2:
                    env.close()
                    conn.close()
                    break
                else:
                    print("bad command: {}".format(command))
        except Exception as e:
            if 'env' in locals() and hasattr(env, 'close'):
                try:
                    env.close()
                    print("closed error")
                except:
                    pass
            conn.send(e)
