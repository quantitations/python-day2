import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

# Set the data folder and the number of papers to scrape (papers are identified by integers)
datapath = Path("federalist_papers")
num_papers = 2

# Retrieve the contents of all the federalist papers from the law school website and save to our data folder
print("Scraping federalist papers 1 through 85...")
for i in range(num_papers):
    print(i+1)
    if i < 9:
        ID = "0" + str(i+1)
    else:
        ID = str(i+1)

    # Request HTML of the page for the i^th paper
    r = requests.get("http://avalon.law.yale.edu/18th_century/fed%s.asp" % ID)
    s = BeautifulSoup(r.text, features="html.parser")  # this is the HTML contents of the returned request
    ps = s.findAll("p")  # this is a list of all paragraph elements of the html
    
    # Save the actual text of the paper as a plain text file
    filename = datapath / (ID + ".txt")
    with open(filename, "w") as f:
        for p in ps:
            t = p.text
            t = t.replace(" Return to the Text", "")
            t = t.replace("Ã\x95", "")
            f.write(t + "\n")
    time.sleep(30)
	
# Get all the paper authors as a list (in the same order as the paper numbers)
print("Scraping author names from webpage table...")
authors = []
r = requests.get("https://www.congress.gov/resources/display/content/The+Federalist+Papers")
s = BeautifulSoup(r.text, features="html.parser")
table = s.find("table", {"class": "table table-bordered"})
rows = table.findAll("tr")


for i, row in enumerate(rows[1:]):
    authors.append(row.findAll("td")[2].text)

# Write the author list as a CSV file
with open(datapath / "authors.csv", "w") as f:
	f.write("\n".join(authors))