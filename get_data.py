import requests
import json
import re

url = f"https://ev.caltech.edu/api/v1/"
sites = ["caltech", "jpl", "office001"]
site_id = "jpl"
endpoints = f"sessions/{site_id}/ts"
api_token = "dFLq4KuknTmi4_yCgnGxDEo3Kk8SgEr_CRljG0RL1cw"
# first page, 25 results of each page
r = requests.get(url+endpoints,auth=(api_token,"")).json()  # username={api_token},password=""
page_data = json.dumps(r)
# remaining pages
last_item = r["_links"]["last"]["href"]
page_num = re.match(f"^{endpoints}\?page=(\d+)$",last_item).group(1)
print(f"total pages: {page_num}")  # 1257, 1346, 68
for i in range(1258, 1346+1, 1):
    page_id = f"?page={i}"
    try:
        r = requests.get(url+endpoints+page_id,auth=(api_token,"")).json()
        page_data = json.dumps(r)
        with open(f"ACN-data/{site_id}/{i}.json","w") as f:
            f.write(page_data)
            print(f"page{i} write done!")
    except:
        print(f"page{i} write wrong!")