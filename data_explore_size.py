import pickle

def get_ais_mes(d):
    all_ais = 0
    for mes in d:
        all_ais += len(mes["FourHot"].todense())
    return all_ais

path = "/Volumes/MNG/data/"
datasets = ["_bh_", "_blt_", "_sk_"]

for ds in datasets:
    print(f"Stats from dataset: {ds}")
    with open(path+"train"+ds+".pcl", "rb") as f:
        train = pickle.load(f)

    with open(path+"val"+ds+".pcl", "rb") as f:
        val = pickle.load(f)

    with open(path+"test"+ds+".pcl", "rb") as f:
        test = pickle.load(f)

    train_ais = get_ais_mes(train)
    val_ais = get_ais_mes(val)
    test_ais = get_ais_mes(test)

    print(f"Train:")
    print(f"-------No. tracks: {len(train)}")
    print(f"-------No. AIS: {train_ais}")

    print(f"Validation:")
    print(f"-------No. tracks: {len(val)}")
    print(f"-------No. AIS: {val_ais}")

    print(f"Test:")
    print(f"-------No. tracks: {len(test)}")
    print(f"-------No. AIS: {test_ais}")


