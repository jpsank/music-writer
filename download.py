import requests
import os
import concurrent.futures


def dl_result(result,idx,out):
    idn = result["id"]
    name = result["name"]

    fname = '{}-{}'.format(idn, name)
    fp = os.path.join(out, fname)

    if not os.path.exists(fp):
        print(" ",idx, fname)

        downloadUrl = "https://bitmidi.com" + result["downloadUrl"]
        r = requests.get(downloadUrl)
        with open(fp, "wb") as f:
            f.write(r.content)


def dl_all(out, page=0):
    r = requests.get("https://bitmidi.com/api/midi/search?pageSize=10000&page={}".format(page))
    j = r.json()
    results = j["result"]["results"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_to_idx = {executor.submit(dl_result, result, i, out): i
                         for i, result in enumerate(results)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            future.result()


def csvify(midi_dir,csv_dir):
    for fname in os.listdir(midi_dir):
        fp1 = os.path.join(midi_dir, fname)
        fp2 = os.path.join(csv_dir ,fname[:-4]+".csv")
        if not os.path.exists(fp2):
            print(fname)
            os.system("midicsv {} {}".format(fp1,fp2))


for i in range(1):
    print("Page number ",i)
    dl_all("data/midi",i)

#csvify("data/midi","data/csv")


