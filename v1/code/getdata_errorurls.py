import xml.etree.ElementTree as ET
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup

ns = {'ns': 'http://data.stortinget.no'}

# Step 1: Collecting case ID's
print("Collecting all cases (sakID's)")
base_url = 'https://data.stortinget.no/eksport/saker?sesjonid=2010-2011'
response = requests.get(base_url)
root = ET.fromstring(response.content)

sak_ids = []
for sak in root.findall('.//ns:sak', namespaces=ns):
    id_element = sak.find('ns:id', namespaces=ns)
    if id_element is not None:
        sak_ids.append(id_element.text)

print(f"Found {len(sak_ids)} cases.")

saks_data = []
nr_of_cases = len(sak_ids)

# Step 2: Collecting information and summaries
print("Collecting information + 'sammendrag' for the cases.")
for i, sakid in enumerate(sak_ids[:nr_of_cases]):
    print(f"Prosesserer sak {i+1}/{nr_of_cases}: {sakid}")

    detalj_url = f'https://data.stortinget.no/eksport/sak?sakid={sakid}'
    detalj_response = requests.get(detalj_url)
    detalj_root = ET.fromstring(detalj_response.content)

    # Collecting title
    tittel_element = detalj_root.find('.//ns:tittel', namespaces=ns)
    tittel = tittel_element.text if tittel_element is not None else 'Ikke oppgitt'

    # Collecting subject words
    emne_liste = [emne.find('ns:navn', namespaces=ns).text for emne in detalj_root.findall('.//ns:emne_liste/ns:emne', namespaces=ns) if emne.find('ns:navn', namespaces=ns) is not None]

    # Collecting summary from the case (fra innstillingen) 
    publikasjoner = detalj_root.findall('.//ns:publikasjon_referanse_liste/ns:publikasjon_referanse', namespaces=ns)
    sammendrag_tekst = None

    for publikasjon in publikasjoner:
        lenke_url = publikasjon.find('ns:lenke_url', namespaces=ns)

        if lenke_url is not None and 'inns' in lenke_url.text:
            full_url = f"https:{lenke_url.text}/?lvl=0"
            #print(full_url)

            try:
                instilling_response = requests.get(full_url)
                html_content = instilling_response.text

                soup = BeautifulSoup(html_content, 'html.parser')
                content_areas = soup.find_all('div', class_='page-publication')

                if len(content_areas) > 0:
                    content_area = content_areas[0]
                    sammendrag_tekst = content_area.get_text(separator='\n', strip=True).replace('\n',' ').replace('\r',' ').replace('  ',' ')
                    print(f"  - Found summary for case number {sakid}")
                    try:
                        heading = soup.find('p', string='vedtak:')
                        vedtak_text= ''
                        for sibling in heading.find_next_siblings('p'):
                            vedtak_text+= sibling.text.replace('\n',' ').replace('\r',' ').replace('  ',' ')+' '
                        print(vedtak_text)
                    except:
                        print('Didnot find accurate block at', full_url)

                else:
                    print(f"  - No summary found for case number {sakid} at {full_url}")

            except requests.exceptions.RequestException as e:
                print(f"  - Could not retrieve summary from {full_url}. Error: {e}")

            time.sleep(1)
            break  # Take only the first relevant publication

    time.sleep(0.5)
