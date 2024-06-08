import math
import os
import os.path
import pickle
import shutil
import time
import warnings
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from numpy import ndarray as arr
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from algos.MAPPO import GR_MAPPO, GR_MAPPOPolicy
from algos.gnn_util import format_training_duration, compute_mcs_edge_adj, \
    compute_disseminated_workload
from algos.ppo_buffer import GraphReplayBuffer

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cur_dir = Path(os.path.dirname(__file__))
models_dir = Path(cur_dir, "model state")
models_dir.mkdir(exist_ok=True)
gma_model = Path(models_dir, "gma")
gma_model.mkdir(exist_ok=True)
results_dir = Path("/Users/tonghaogang/Project/GatMicroservice/simulation/Evaluation", "Results")
results_dir.mkdir(exist_ok=True, parents=True)
pict_dir = Path("/Users/tonghaogang/Project/GatMicroservice/simulation/Evaluation", "Pictures")
pict_dir.mkdir(exist_ok=True, parents=True)

MS = int(1e3)

KB = int(8 * 1e3)
MB = int(8 * 1e6)
GB = int(8 * 1e9)

MHZ = int(1e6)
GHZ = int(1e9)

tick_size = 20
label_size = 20
legend_size = 15
title_size = 20
text_size = 15


class Metrics:
    """
    This class stores metric values calculated based on Shannon's formula and queueing theory,
    such as the calculation of response time.
    """
    # In the air, the speed of electromagnetic waves is approximately 300,000 km/s
    SpeedOfElectro = 3 * int(1e5)

    LatencyOfCloud = 50

    @staticmethod
    def shannon_formula_db(B, SN):
        """
           Transmission delay: Calculate channel capacity based on Shannon's formula, with SNR input in dB
           Args:
               B (float): Bandwidth (Hz)
               SN (float): Signal-to-noise ratio (dB)

           Returns:
               Channel capacity (Mbps)
        """
        # Convert dB to linear scale, referring to the SNR conversion formula
        _SN = np.log10(SN / 10)
        return B * np.log2(1 + _SN)

    @staticmethod
    def shannon_formula(B, S, N):
        """
            Transmission delay: Calculate channel capacity based on Shannon's formula
            Args:
                B (float): Bandwidth (Hz)
                S (float): Signal power (W)
                N (float): Noise power (W)

            Returns:
                Channel capacity (Mbps)
        """
        return B * np.log2(1 + S / N)

    @staticmethod
    def send_delay(data_size, channel_capacity):
        """
        Transmission delay, the time required to push the data packet onto the link
        Args:
            data_size: Data size (MB)
            channel_capacity: Channel capacity (Mbps)

        Returns:
            Transmission delay (ms)
        """
        return round((data_size / channel_capacity) * MS)

    @staticmethod
    def process_time(u_lambda, mu=100, instance=1):
        """
            Processing delay: Calculate the average processing time for each request based on the M/M/1 model of queueing theory,
            using a round-robin policy to select the instance for processing the request.
            Args:
                mu: Departure rate
                u_lambda: Poisson distribution parameter, arrival rate
                instance: Number of instances where the service is deployed
            Returns:
                Processing time (ms)
        """
        omega = torch.nn.init.trunc_normal_(torch.zeros((1,)), a=1, b=1.08).item()
        # omega = torch.nn.init.trunc_normal_(torch.zeros((1,)), a=1, b=1.5).item()
        # init_omega = np.random.uniform(2, 3)
        prc_time = 0
        is_init = False
        threshold_coefficient = int(0.93 * mu)
        # if instance == 1:
        #     is_init = True
        #     instance = 2
        while math.ceil(u_lambda / instance) >= threshold_coefficient:
            prc_time += round(omega * (1 / (mu - threshold_coefficient)) * MS)
            u_lambda -= threshold_coefficient
        prc_time += round(omega * (1 / (mu - math.ceil(u_lambda / instance))) * MS, 3)
        return round(prc_time, 3)

    @staticmethod
    def propagation_time(distance):
        """
        Signal transmission delay in the medium (ms), this delay is only related to distance
        Args:
            distance: Transmission distance (km)

        Returns:
            Transmission delay (ms)
        """
        return round((distance / Metrics.SpeedOfElectro) * MS)


class Application:
    cur_dir = Path(os.path.dirname(__file__))
    sv_dir = Path(cur_dir, "mcs")
    sv_dir.mkdir(parents=True, exist_ok=True)
    graph_file = Path(cur_dir, "graph_0.pkl")

    def __init__(self, is_gen=False):
        self.n_microservice = None
        self.G = nx.DiGraph()
        self.microservices = []
        self.adj = None
        self.cycles = []
        if not is_gen:
            try:
                self.load_graph()
            except FileNotFoundError as e:
                self.gen_graph()
            except EOFError:
                self.gen_graph()
        else:
            self.gen_graph()

    def gen_graph(self, adj=None, n_microservice=8, is_save=True, is_draw=False):
        self.prepare_files()
        self.G.clear()
        self.microservices.clear()
        if adj is None:
            self.n_microservice = n_microservice
            self.adj = torch.randint(0, 2, (self.n_microservice, self.n_microservice)).numpy()
        else:
            self.n_microservice = len(adj)
            self.adj = adj

        for i in range(self.n_microservice):
            self.microservices.append(Microservice(i))
        for i in range(self.n_microservice):
            self.G.add_node(i, color="red")
            for j in range(self.n_microservice):
                if i == j:
                    continue
                if self.adj[i, j] == 1:
                    self.G.add_edge(i, j, weight=3, comment="ok")
                    self.microservices[i].children.append(j)

        self.remove_cycle()
        if is_save:
            self.save_graph()
        if is_draw:
            self.draw_graph()

    def prepare_files(self):
        files = []
        mx_files = 10
        for f in Application.cur_dir.iterdir():
            if Path.is_file(f) and f.name.startswith("graph"):
                files.append(f)
        files.sort(key=lambda i: int(i.name.split('_')[-1].split(".")[0]))
        sv_files = []
        for f in Application.sv_dir.iterdir():
            if Path.is_file(f) and f.name.startswith("graph"):
                sv_files.append(f)
        sv_files.sort(key=lambda i: int(i.name.split('_')[-1].split(".")[0]))
        for f in sv_files[:-1 * mx_files]:
            if f.is_file():
                print(str(f))
                f.unlink()
        sv_files = sv_files[-1 * mx_files:]
        for i in range(len(sv_files)):
            shutil.move(Path(Application.sv_dir, sv_files[i]),
                        Path(Application.sv_dir, "graph_{}.pkl".format(i)))
        for i in range(len(files)):
            shutil.move(Path(Application.cur_dir, files[i]),
                        Path(Application.sv_dir, "graph_{}.pkl".format(len(sv_files) + i)))

    def remove_cycle(self):
        self.find_cycles()
        while len(self.cycles) >= 1:
            for cycle in self.cycles:
                edge0, edge1 = cycle[-2], cycle[-1]
                try:
                    self.G.remove_edge(edge0, edge1)
                    self.adj[edge0, edge1] = 0
                    self.microservices[edge0].children.remove(edge1)
                except nx.exception.NetworkXError:
                    print("error")
                break
            self.cycles = []
            self.find_cycles()

    def find_cycles(self):
        for m in range(self.n_microservice):
            self._dfs([m])

    def _dfs(self, path):
        for m in self.microservices[path[-1]].children:
            if m in path:
                self.cycles.append(path[path.index(m):] + [m])
                continue
            self._dfs(path + [m])

    def save_graph(self):
        with Application.graph_file.open("wb") as f_path:
            pickle.dump(self.adj, f_path)
            pickle.dump(self.G, f_path)
            pickle.dump(self.microservices, f_path)
            pickle.dump(self.n_microservice, f_path)

    def load_graph(self, is_draw=False):
        try:
            with Application.graph_file.open("rb") as f_path:
                self.adj = pickle.load(f_path)
                self.G = pickle.load(f_path)
                self.microservices = pickle.load(f_path)
                self.n_microservice = pickle.load(f_path)
        except FileNotFoundError:
            f = list(Application.sv_dir.iterdir())
            if len(f) <= 0:
                raise FileNotFoundError
            f.sort(key=lambda i: int(i.name.split('_')[-1].split(".")[0]))
            with f[-1].open("rb") as f_path:
                self.adj = pickle.load(f_path)
                self.G = pickle.load(f_path)
                self.microservices = pickle.load(f_path)
                self.n_microservice = pickle.load(f_path)
        if is_draw:
            self.draw_graph()

    def draw_graph(self):
        node_colors = [self.G.nodes[node]["color"] for node in self.G.nodes]

        pos = nx.circular_layout(self.G)
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            font_size=12,
            font_color="white",
            font_weight="bold",
            arrows=True,
        )
        edge_labels = {(u, v): d["weight"] for u, v, d in self.G.edges(data=True)}
        # nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=15)

        edge_labels = {
            (u, v): data["comment"] for u, v, data in self.G.edges(data=True)
        }

        plt.title("1122")
        plt.show()


class Cluster:
    cluster = {}
    s_m = None

    @staticmethod
    def init_cluster(n_servers, app: Application):
        Cluster.s_m = np.zeros((n_servers, len(app.microservices)))
        Cluster.s_m[-1] = 1
        # for index in range(n_servers - 1):
        #     Cluster.cluster[index] = Cluster.Server(app, server_index=index)
        # Cluster.cluster[-1] = Cluster.Server(app, 1, server_index=-1)

    @staticmethod
    def reset_cluster():
        Cluster.s_m = np.zeros_like(Cluster.s_m)
        Cluster.s_m[-1] = 1

    @staticmethod
    def deploy_mcs(server_ind, mcs_name: int, add_ins):
        min_svc = np.argmin(Cluster.s_m[server_ind])
        max_svc = np.argmax(Cluster.s_m[server_ind])
        if mcs_name == max_svc and Cluster.s_m[server_ind, max_svc] - Cluster.s_m[server_ind, min_svc] >= 10:
            Cluster.s_m[server_ind, max_svc] = min(Cluster.s_m[server_ind, max_svc],
                                                   Cluster.s_m[server_ind, max_svc] + add_ins)
            return
        if server_ind < Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 15 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1
        if server_ind == Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 100 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1

        Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name] + add_ins, 0)
        if server_ind == len(Cluster.s_m) - 1:
            Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name], 1)

    @staticmethod
    def deploy_mcs_mask(server_ind, mcs_name: int, add_ins, adj):
        svcNotIngraph = np.where(np.all(adj == 0, axis=1))[0].astype(np.int32)
        min_svc = np.argmin(Cluster.s_m[server_ind])
        max_svc = np.argmax(Cluster.s_m[server_ind])
        if mcs_name == max_svc and Cluster.s_m[server_ind, max_svc] - Cluster.s_m[server_ind, min_svc] >= 10:
            Cluster.s_m[server_ind, max_svc] = min(Cluster.s_m[server_ind, max_svc],
                                                   Cluster.s_m[server_ind, max_svc] + add_ins)
            return
        if server_ind < Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 15 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1
        if server_ind == Cluster.s_m.shape[0] - 1 and np.sum(Cluster.s_m[server_ind]) >= 100 \
                and add_ins >= 0:
            Cluster.s_m[server_ind, np.argmax(Cluster.s_m[server_ind])] = 1

        Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name] + add_ins, 0)
        if server_ind == len(Cluster.s_m) - 1:
            Cluster.s_m[server_ind, mcs_name] = max(Cluster.s_m[server_ind, mcs_name], 1)
        for svc in svcNotIngraph:
            Cluster.s_m[:, svc] = 0

    # @staticmethod
    # def reset_cluster():
    #     Cluster.s_m = np.zeros_like(Cluster.s_m)
    #     Cluster.s_m[-1] = 1
    #     for k in Cluster.cluster:
    #         Cluster.cluster[k].svc2instance = {}
    #     for k in Cluster.cluster[-1].svc2instance:
    #         Cluster.cluster[-1].svc2instance[k] = 1

    def _init_edge(self):
        self.server_type = 0
        self.limited_instance = np.random.randint(3, 15)
        self.memory = np.random.randint(400, 1100) * MB
        self.cpu = np.random.randint(1, 4)
        self.bandwidth = np.random.randint(100, 200) * MHZ

    def _init_cloud(self):
        self.server_id = -1
        self.server_type = 1
        self.limited_instance = float('inf')
        self.memory = 1 * GB
        self.cpu = 4
        self.bandwidth = 2 * GB

    @staticmethod
    def remove_server(i):
        Cluster.s_m = np.concatenate([Cluster.s_m[:i, ...], Cluster[i + 1:, ...]])

    @staticmethod
    def add_server():
        new_server = np.zeros((Cluster.s_m.shape[0], Cluster.s_m.shape[1]))
        Cluster.s_m = np.concatenate([Cluster.s_m[:-1, ...], new_server,
                                      Cluster.s_m[-1, ...]])


class Microservice:
    def __init__(self, name):
        # Index of the service
        self.name = name
        # Stores the index of the service
        self.children = []

    def get_prc_time(self, src_server, n_user):
        total_time = 0
        # 1. If an instance of the target service exists on the target server, process it directly on the target server
        if Cluster.s_m[src_server, self.name] > 0:
            total_time += Metrics.process_time(n_user, instance=Cluster.s_m[src_server, self.name])
        else:
            # 2. Otherwise, traverse the edge server set to find an available server for processing
            available_server = list(*np.where(Cluster.s_m[:-1, self.name] > 0))
            if len(available_server) > 0:
                total_time += Metrics.process_time(n_user,
                                                   instance=Cluster.s_m[available_server[0], self.name])
            # 3. If not, handle it on the cloud server
            # Round-trip propagation delay of the cloud server
            else:
                t_prop = 2 * torch.nn.init.trunc_normal_(tensor=torch.zeros(1, ), mean=11, std=2, a=10, b=15).item()
                if Cluster.s_m[-1, self.name] <= 0:
                    Cluster.s_m[-1, self.name] = 1
                total_time += t_prop + Metrics.process_time(n_user, instance=Cluster.s_m[-1, self.name])
        return total_time


class UserGroup:
    users = {}

    class User:
        def __init__(self, longitude=-1, latitude=-1, server=-1):
            # Indicates that the user is not near any edge server;
            # requests will be directly scheduled to the cloud server
            self.location = -1
            self.longitude = longitude
            self.latitude = latitude
            # Indicates which server range the user is currently in; -1 represents the cloud server range
            self.server = server
            UserGroup.users[server] = UserGroup.users.get(server, [])
            UserGroup.users[server].append(self)

    def __init__(self):
        pass

    @staticmethod
    def init_users(n_users: int = 200):
        # for _ in range(n_users):
        #     server = np.random.choice(list(Cluster.cluster.keys()), 1).item()
        #     UserGroup.User(server=server)
        pass

    def send_request(self, mcs, n_user, app: Application, src_server_id=-1, path=None, mcs_response_time=None):
        return self.get_process_time(mcs, n_user, app, src_server_id, path, mcs_response_time=mcs_response_time)

    def get_process_time(self, mcs: Microservice, n_user, app: Application, src_server_id=-1, path=None, sub_path=None,
                         mcs_response_time=None):
        assert mcs_response_time is not None and isinstance(mcs_response_time, dict), "must pass dict mcs_response_time"
        if sub_path is None:
            sub_path = []
        if path is None:
            path = []
        sub_path.append(mcs.name)
        prc_time = 0
        # Traverse to nodes with a degree of 0, which are leaf nodes, and record the sub-call path
        if len(mcs.children) == 0:
            path.append(sub_path)
        for m in mcs.children:
            prc_time += self.get_process_time(app.microservices[m], n_user, app, src_server_id, path, [] + sub_path,
                                              mcs_response_time=mcs_response_time)
        cur_mcs_time = mcs.get_prc_time(src_server_id, n_user)
        prc_time += cur_mcs_time
        mcs_response_time[mcs.name] = mcs_response_time.get(mcs.name, []) + [cur_mcs_time]
        return round(prc_time, 3)


class EdgeCloudSim:
    traffic_pth = Path(cur_dir, "dataset", "traffic_data_simulation.npy")
    data_mcs_pth = Path(cur_dir, "dataset", "traffic_mcs_data_simulation.npy")
    traffic_mcs_pth = Path(cur_dir, "dataset", "traffic_data_simulation.npz")
    data_pics_dir = Path(cur_dir, "dataset", "pics", 'mcs_traffic')
    data_pics_dir.mkdir(parents=True, exist_ok=True)
    [i.unlink() for i in data_pics_dir.iterdir()]

    # Model training file path integrity check
    train_data_pth = Path(gma_model, "training data")
    train_data_pth.mkdir(parents=True, exist_ok=True)
    gma_train_pth = Path(train_data_pth, "GMA data")
    gma_train_pth.mkdir(parents=True, exist_ok=True)

    def __init__(self, n_server=8, n_user=200, is_gen=False, if_mask=False):
        self.is_continuous = False
        self.mode = "train"
        self.algotype = "gat"
        self.application = Application(is_gen)
        self.num_mcs = len(self.application.microservices)
        self.n_server = n_server
        self.n_user = n_user
        self.t_s_m = np.load(EdgeCloudSim.data_mcs_pth)[:, :self.n_server, :]
        self.t_s_m[..., -2] += 300
        self.traffic = np.array(([int(i.strip()) for i in list(Path(self.get_dataset_dir(),
                                                                    "flows.txt").open("r"))]))
        self.reward_pth = Path(EdgeCloudSim.gma_train_pth, "gma_reward.pt")
        self.train_state_pth = Path(EdgeCloudSim.gma_train_pth, "gma_object.pt")
        self.t = 0
        self.action_set = [-1, 0, 1]
        self.actions = self.get_actions()
        self.if_mask = if_mask
        self.action_space = np.arange(-2, 3)
        self.entrance_mcs = np.load("/Users/tonghaogang/Project/Paper-Scalable/simulation/dataset/access_data.npy")
        self.actual_workload = compute_disseminated_workload(self.application.adj,
                                                             self.entrance_mcs,
                                                             self.t_s_m)
        self.msc_edge_adj = compute_mcs_edge_adj(self.application.adj,
                                                 self.entrance_mcs,
                                                 self.t_s_m)
        Cluster.init_cluster(n_server, self.application)
        UserGroup.init_users(self.n_user)

    def get_actions(self):
        def _get_actions(d=self.num_mcs):
            re = []
            if d == 0:
                return []
            for i in self.action_set:
                acts = _get_actions(d - 1)
                if len(acts) > 0:
                    re.extend([[i] + j for j in acts])
                else:
                    re.append([i])
            return re

        return _get_actions()

    def get_dataset_dir(self):
        return Path(os.path.dirname(__file__), "dataset")

    def send_dataset_workload(self, t=0):
        """
        Send traffic according to the dataset while obtaining feedback from the environment

        @param t: Current time step of the experiment
        @return: Weighted average response time of the cluster,
        weighted response times of each server,
        average response time of the cluster, response times of each server
        """
        # user request response time
        user_res = np.zeros((self.n_server, self.num_mcs))
        total_res = np.zeros((self.n_server, self.num_mcs))
        # Response times of m services on n servers at time t
        svc_res = np.zeros((self.n_server, self.num_mcs))
        total_users = 0
        for s in range(self.n_server):
            mcs_response_time = {}
            entry_service = np.where(self.entrance_mcs[t, s] == 1)[0]
            for m in entry_service:
                mcs = self.application.microservices[m]
                path = []
                n_mcs_user = self.t_s_m[t, s, m] + 400
                total_users += n_mcs_user
                if s == self.n_server - 1:
                    s = -1
                res = UserGroup().send_request(mcs, n_mcs_user, self.application, s, path, mcs_response_time)
                _path = sorted(set([tuple(i) for i in path]), key=lambda i: path.index(list(i)))
                total_res[s, m] = res * n_mcs_user
                user_res[s, m] = res
            for i in range(svc_res.shape[-1]):
                svc_res[s, i] = np.mean(mcs_response_time.get(i, 0))
        avg_res = np.round(np.sum(total_res) / total_users)
        avg_user_res = np.mean(user_res)
        return avg_res, total_res, avg_user_res, user_res, svc_res

    def traffic(self, slot=200):
        period = 10
        users2mcs = {}
        for t in range(slot):
            for m in self.application.microservices:
                users2mcs[m] = users2mcs.get(m, [])
                if t == 0:
                    users2mcs[m].append(20)
                else:
                    if (t // period) & 1 == 0:
                        print(t // period)
                        users2mcs[m].append(users2mcs[m][-1] + np.random.randint(1, 10))
                    else:
                        users2mcs[m].append(users2mcs[m][-1] + np.random.randint(-9, -1))
        return users2mcs

    def get_state(self, t):
        """
        Construct features: current load and number of instances for each microservice,
        and current load and number of deployed instances on each server.
        Args:
            t: current time step

        Returns:
            numpy array:  server obs, svc obs
        """
        m = self.num_mcs
        s = self.n_server
        user_server = np.sum(self.t_s_m[t], axis=-1, keepdims=True)
        actual_workload = self.actual_workload[t, :s, :m]
        avg_total_res, total_res, avg_user_res, user_res, svc_res = self.send_dataset_workload(t)
        avg_res_all_servers = np.repeat(np.array(avg_user_res)[np.newaxis, np.newaxis], s, axis=0)
        workload_on_servers = np.concatenate([self.t_s_m[t], np.zeros_like(self.t_s_m[t])[..., :m - 5]], axis=-1)
        server_obs = np.concatenate([workload_on_servers, Cluster.s_m, avg_res_all_servers,
                                     user_res, svc_res, actual_workload], axis=-1)

        workload_on_svc = actual_workload.T
        total_svc_workload = np.sum(workload_on_svc, axis=-1, keepdims=True)
        avg_svc_workload = np.mean(workload_on_svc, axis=-1, keepdims=True)
        total_svc_ins = np.sum(Cluster.s_m.T, axis=-1, keepdims=True)
        avg_svc_ins = np.mean(Cluster.s_m.T, axis=-1, keepdims=True)
        avg_svc_res = np.mean(svc_res.T, axis=-1, keepdims=True)
        svc_obs = np.concatenate([total_svc_workload, avg_svc_workload, total_svc_ins, avg_svc_ins, avg_svc_res],
                                 axis=-1)

        return server_obs, svc_obs, user_res

    def reward(self, user_res, t=0):
        server_res = np.mean(user_res, axis=-1)
        server_ins = np.sum(Cluster.s_m, axis=-1)
        server_reward = []
        sla = 50
        limit_rsc_edge = 8
        cloud_rsc_limit = 37

        print("instance distribution: {}".format(Cluster.s_m))
        # print("accessed services:".format(self.entrance_mcs[t]))
        # print("workload distribution: {}".format(self.actual_workload[t]))

        for i in range(self.n_server):
            res_time = server_res[i]
            total_ins = server_ins[i]
            reward_res = sla - res_time
            limit_rsc = cloud_rsc_limit if i == self.n_server - 1 else limit_rsc_edge
            reward_ins = limit_rsc - total_ins
            workload_on_server = self.actual_workload[t, i][np.newaxis, :]
            resource_pattern = Cluster.s_m[i][np.newaxis, :]
            pearsonr_score = pearsonr(self.actual_workload[t, i], Cluster.s_m[i])[0]
            cosine_score = cosine_similarity(workload_on_server, resource_pattern).item()
            pattern_score = 0.7 * cosine_score + 0.3 * pearsonr_score if not np.isnan(pearsonr_score) else cosine_score

            server_reward.append(0.12 * reward_ins + 0.75 * reward_res)

        return np.array(server_reward)

    def deploy_mcs(self, server_ind, mcs_name: int, add_ins):
        num_server = len(Cluster.s_m)
        # Set resource thresholds for edge servers and cloud servers.
        total_rsc = 10 if server_ind < num_server - 1 else 50
        used_rsc = np.sum(Cluster.s_m[server_ind])
        remain_rsc = max(total_rsc - used_rsc, 0)
        if add_ins > 0 >= remain_rsc:
            return
        Cluster.s_m[server_ind, mcs_name] += add_ins
        Cluster.s_m[server_ind, mcs_name] = max(0, Cluster.s_m[server_ind, mcs_name])

    def reset(self):
        Cluster.reset_cluster()
        obs, svc_obs, user_res = self.get_state(0)
        agent_id = np.reshape(np.arange(0, self.n_server), (1, -1, 1))
        adj = np.ones((self.n_server, self.n_server))
        return obs[np.newaxis, ...], svc_obs, agent_id, obs[:, np.newaxis, :].repeat(self.n_server, axis=1), adj

    def step(self, actions: Union[torch.Tensor, np.ndarray], t=0):
        action_step = self.action_space[actions]
        original_shape = action_step.shape
        action_step_resize = np.reshape(action_step, (-1, original_shape[-2], original_shape[-1]))
        for i in range(action_step_resize.shape[0]):
            for j in range(action_step_resize.shape[1]):
                for k in range(action_step_resize.shape[2]):
                    self.deploy_mcs(j, k, action_step_resize[i, j, k])
        obs, svc_obs, user_res = self.get_state(t)
        agent_id = np.reshape(np.arange(0, self.n_server), (1, -1, 1))
        adj = np.ones((self.n_server, self.n_server))
        dones = np.full((1, self.n_server), t >= 199)
        return (obs, svc_obs, agent_id, obs, adj,
                self.reward(user_res, t)[np.newaxis, :, np.newaxis], dones, None)


def _t2n(x):
    return x.detach().cpu().numpy()


class GMPERunner:
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    dt = 0.1

    def __init__(self, num_servers=8, num_services=5):
        self.n_rollout_threads = 1
        self.envs = EdgeCloudSim(n_server=num_servers)
        self.svc_adj = self.envs.application.adj[np.newaxis, ...].repeat(num_servers, axis=0)
        self.num_env_steps = 10000
        self.episode_length = 200
        self.all_args = None
        self.num_agents = num_servers
        self.num_services = num_services
        self.use_linear_lr_decay = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = GR_MAPPOPolicy(26, num_services, self.svc_adj)
        self.trainer = GR_MAPPO(self.policy, device=self.device)
        self.buffer = GraphReplayBuffer(26, self.num_agents, self.num_services)
        self.recurrent_N = 1
        self.use_centralized_V = True
        self.hidden_size = 64
        self.save_dir = "/Users/tonghaogang/Project/Paper-Scalable/simulation/模型状态保存/InfoPPO"
        self.training_log_dir = Path(self.save_dir, "training_log")
        self.training_log_dir.mkdir(parents=True, exist_ok=True)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.restore()

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        try:
            policy_actor_state_dict = torch.load(
                str(self.save_dir) + "/actor.pt", map_location=torch.device("cpu")
            )
            self.policy.actor.load_state_dict(policy_actor_state_dict)
        except FileNotFoundError:
            print("actor model not saved")
        try:
            policy_critic_state_dict = torch.load(
                str(self.save_dir) + "/critic.pt", map_location=torch.device("cpu")
            )
            self.policy.critic.load_state_dict(policy_critic_state_dict)
        except FileNotFoundError:
            print("critic model not saved")

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def run(self):
        self.warmup()

        start_time = time.time()
        episodes = 500
        rewards_log = []
        # This is where the episodes are actually run.
        for episode in range(episodes):
            print("Episode {}/{}".format(episode + 1, episodes))
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            rewards_episode = []
            # Reset the environment at the beginning.
            self.envs.reset()
            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obs reward and next obs
                obs, svc_obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    torch.tensor(actions_env)
                )
                rewards_episode.append(rewards)

                data = (
                    obs,
                    svc_obs,
                    agent_id,
                    node_obs,
                    adj,
                    self.envs.msc_edge_adj[step + 1],
                    agent_id,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)
            rewards_log.append(np.array(rewards_episode).sum())

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                    (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % 5 == 0 or episode == episodes - 1:
                self.save()
                torch.save(np.array(rewards_log), Path(self.training_log_dir, "rewards.pt"))

            # # log information
            # if episode % self.log_interval == 0:
            #     end = time.time()
            #
            #     env_infos = self.process_infos(infos)
            #
            #     avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
            #     train_infos["average_episode_rewards"] = avg_ep_rew
            #     print(
            #         f"Average episode rewards is {avg_ep_rew:.3f} \t"
            #         f"Total timesteps: {total_num_steps} \t "
            #         f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
            #     )
            #     self.log_train(train_infos, total_num_steps)
            #     self.log_env(env_infos, total_num_steps)
            #
            # # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)
        end_time = time.time()
        print("training done, total time taken:", format_training_duration(end_time - start_time))

    def warmup(self):
        # reset env
        obs, svc_obs_, agent_id, node_obs, adj = self.envs.reset()

        # replay buffer
        # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
        share_obs = obs.reshape(self.n_rollout_threads, -1)
        # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
        share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
        share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
        # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
        share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
            self.num_agents, axis=1
        )
        svc_obs = svc_obs_[np.newaxis, np.newaxis, ...].repeat(self.num_agents, axis=1)

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.svc_obs[0] = svc_obs.copy()
        self.buffer.node_obs[0] = node_obs.copy()
        self.buffer.adj[0] = adj.copy()
        self.buffer.svc_adj[0] = self.envs.msc_edge_adj[0].copy()
        self.buffer.agent_id[0] = agent_id.copy()
        self.buffer.share_agent_id[0] = share_agent_id.copy()

    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.node_obs[step]),
                                            np.concatenate(self.buffer.adj[step]),
                                            np.concatenate(self.buffer.agent_id[step]),
                                            np.concatenate(self.buffer.share_agent_id[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            svc_obs=np.concatenate(self.buffer.svc_obs[step]),
                                            svc_adj=np.concatenate(self.buffer.svc_adj[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions,
        )

    def insert(self, data):
        (
            obs,
            svc_obs,
            agent_id,
            node_obs,
            adj,
            svc_adj,
            agent_id,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # if centralized critic, then shared_obs is concatenation of obs from all agents
        if self.use_centralized_V:
            # TODO stack agent_id as well for agent specific information
            # (n_rollout_threads, n_agents, feats) -> (n_rollout_threads, n_agents*feats)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*feats) -> (n_rollout_threads, n_agents, n_agents*feats)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # (n_rollout_threads, n_agents, 1) -> (n_rollout_threads, n_agents*1)
            share_agent_id = agent_id.reshape(self.n_rollout_threads, -1)
            # (n_rollout_threads, n_agents*1) -> (n_rollout_threads, n_agents, n_agents*1)
            share_agent_id = np.expand_dims(share_agent_id, 1).repeat(
                self.num_agents, axis=1
            )
        else:
            share_obs = obs
            share_agent_id = agent_id

        self.buffer.insert(share_obs, obs, node_obs, adj, agent_id, share_agent_id, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, svc_obs=svc_obs, svc_adj=svc_adj)

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                     np.concatenate(self.buffer.node_obs[-1]),
                                                     np.concatenate(self.buffer.adj[-1]),
                                                     np.concatenate(self.buffer.share_agent_id[-1]),
                                                     np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]),
                                                     np.concatenate(self.buffer.svc_obs[-1]),
                                                     np.concatenate(self.buffer.svc_adj[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    def eval_cma(self):
        eval_rnn_states = np.zeros(
            (1, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (1, self.num_agents, 1), dtype=np.float32
        )
        obs, svc_obs, agent_id, node_obs, adj = self.envs.reset()
        res = []
        pods = []
        pattern_score = []
        for times in range(7):
            res_cur = []
            pods_cur = []
            pattern_score_cur = []
            for eval_step in range(300):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    node_obs,
                    np.ones((self.num_agents, self.num_agents, self.num_agents)),
                    np.concatenate(agent_id),
                    np.concatenate(eval_rnn_states),
                    np.concatenate(eval_masks),
                    svc_obs=svc_obs[np.newaxis, :].repeat(self.num_agents, 0),
                    svc_adj=self.envs.msc_edge_adj[eval_step]
                )
                obs, svc_obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(
                    torch.tensor(eval_action)
                )
                workload_on_server = np.sum(self.envs.actual_workload[eval_step], 0, keepdims=True)
                mask = workload_on_server != 0
                workload_on_server[mask] = workload_on_server[mask] / np.min(workload_on_server)
                resource_pattern = np.sum(Cluster.s_m, 0, keepdims=True)
                cosine_score = cosine_similarity(workload_on_server, resource_pattern).item()
                pattern_score_cur.append(cosine_score)
                res_cur.append(self.envs.send_dataset_workload(eval_step)[2])
                pods_cur.append(Cluster.s_m.sum())
                obs = np.array(obs)[np.newaxis, ...]
                node_obs = np.array(node_obs)[np.newaxis, ...].repeat(self.num_agents, 0)
                eval_rnn_states = np.array(
                    np.split(_t2n(eval_rnn_states), 1)
                )
                eval_rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (1, self.num_agents, 1), dtype=np.float32
                )
                eval_masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )
            res.append(res_cur)
            pods.append(pods_cur)
            pattern_score.append(pattern_score_cur)
        res = np.array(res)
        pods = np.array(pods)
        pattern_score = np.array(pattern_score)


if __name__ == '__main__':
    GMPERunner().train()
    GMPERunner().eval_cma()
