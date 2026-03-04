import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClKmeans:
    def __init__(self, features, device_names, n_clusters=3):
        self.features = features
        self.device_names = device_names
        self.n_clusters = n_clusters

    def k_means(self):
        """
        features     : shape (N, F)
        device_names : list of length N
        n_clusters   : number of power levels

        return:
        {
            "level 1": [device weakest → strongest],
            "level 2": [...],
            ...
        }
        """

        assert len(self.features) == len(self.device_names), "Mismatch input size"

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)

        # KMeans clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=42
        )
        labels = kmeans.fit_predict(X_scaled)

        # Compute power score
        power_score = X_scaled.sum(axis=1)

        # Group devices by cluster-
        cluster_map = {}
        for name, label, score in zip(self.device_names, labels, power_score):
            cluster_map.setdefault(label, []).append((name, score))

        # 5. Sort clusters by average power
        cluster_order = sorted(
            cluster_map.keys(),
            key=lambda c: np.mean([s for _, s in cluster_map[c]])
        )

        # Build result dict
        result = {}
        for i, c in enumerate(cluster_order, start=1):
            # sort devices inside cluster
            devices_sorted = sorted(
                cluster_map[c],
                key=lambda x: x[1]
            )
            result[f"level {i}"] = [d[0] for d in devices_sorted]

        return result

    def run(self):
        if self.n_clusters == 1:
            return self.k_means()
            # print("The number of cluster is 1")
        elif self.n_clusters > len(self.features):
            print("[Ops the number of cluster > length of features ")
        else:
            return self.k_means()


class Clustering:
    # input :

    # lst_devices[0, [0, 'c74b1a5f-5bc6-4946-992b-8f97f5826469'], [0, 'f3b612ea-5a9d-485f-be90-b199e0e1d0bd']]

    # data_clients
    # {'c74b1a5f-5bc6-4946-992b-8f97f5826469': {
    #     'device': {'Total Ram': 16, 'Total Storage': 840, 'Internet': 4000, 'Core': 64}, 'stage': 1},
    #  'f3b612ea-5a9d-485f-be90-b199e0e1d0bd': {
    #      'device': {'Total Ram': 48, 'Total Storage': 420, 'Internet': 2000, 'Core': 32}, 'stage': 2}}

    # return
    # dict with key : uuid and value : cluster_id
    def __init__(self , lst_devices , data_clients , n_cluster ):
        self.lst_devices = lst_devices
        self.data_clients = data_clients
        self.n_cluster = n_cluster
        self.dict_res = {}   # dict store each str uuid correspond to cluster

    def extract_device_info(self , dict_devices ) :    # return list info
        lst_data = []
        for values in dict_devices.values() :
            lst_data.append(values)
        return  lst_data

    def run(self):

        for stage in range(1 , len(self.lst_devices) ):
            id_names = []
            features = []
            for client_id in self.lst_devices[stage]:
                if client_id != 0  :
                    id_names.append(client_id)
                    print(f'check client id {client_id}')
                    features.append(self.extract_device_info(self.data_clients[client_id]['device']))

            print(f'stage {stage}')
            print(f'id names {id_names}')
            print(f'features {features}')

            cluster = ClKmeans(
                features=features,
                device_names=id_names,
                n_clusters=self.n_cluster
            )

            res = cluster.run()
            # RES OF CLUSTERING
            # {'level 1': ['6944d2d1-f8c2-43a6-a023-d5fd3e5727c0'], 'level 2': ['1b084d14-c818-49cf-90a9-3157803fd32d']}

            for level in res.keys():
                for client_id in res[level]:
                    if stage == 1 :
                        self.dict_res[client_id] = int(level[-1])
                    else :
                        self.dict_res[client_id] = self.n_cluster -int(level[-1]) + 1

        return self.dict_res



#  Example feature matrix (N=5 devices, F=3 features)
# features = np.array([
#     [200, 150, 100],   # Device A (weak)
#     [250, 180, 120],   # Device B
#     [800, 700, 650],   # Device C (strong)
#     [850, 720, 680],   # Device D (strongest)
#     [400, 300, 250],   # Device E (medium)
# ])
#
# # Device names (must match number of rows)
# device_names = [
#     "Device_A",
#     "Device_B",
#     "Device_C",
#     "Device_D",
#     "Device_E"
# ]