import requests
import os
import concurrent.futures


def dl_result(result, out):
    idn = result["id"]
    name = result["name"]

    fname = '{}-{}'.format(idn, name)
    fp = os.path.join(out, fname)

    if not os.path.exists(fp):
        download_url = "https://bitmidi.com" + result["downloadUrl"]
        r = requests.get(download_url)
        with open(fp, "wb") as f:
            f.write(r.content)

        return fname


def dl_all(out, page=0):
    r = requests.get("https://bitmidi.com/api/midi/search?pageSize=10000&page={}".format(page))
    j = r.json()
    results = j["result"]["results"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(dl_result, result, out) for result in results]
        for future in concurrent.futures.as_completed(futures):
            if res := future.result():
                print(res)


def csvify(midi_dir,csv_dir):
    for fname in os.listdir(midi_dir):
        fp1 = os.path.join(midi_dir, fname)
        fp2 = os.path.join(csv_dir, fname[:-4]+".csv")
        if not os.path.exists(fp2):
            print(fname)
            os.system("midicsv {} {}".format(fp1,fp2))


for p in range(1):
    print("Page number", p)
    dl_all("data/midi/downloaded", p)

# csvify("data/midi","data/csv")

