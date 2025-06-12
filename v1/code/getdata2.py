### Pulling 'Tilråding' in addition to the whole sammendrag. Saving in 'data2.csv'


import xml.etree.ElementTree as ET
import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
import re

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
                    vedtak_text= ''
                    try:
                        heading = soup.find('p', string='vedtak:')
                        for sibling in heading.find_next_siblings('p'):
                            vedtak_text+= sibling.text.replace('\n',' ').replace('\r',' ').replace('  ',' ')+' '
                        print(vedtak_text)
                        
                        if vedtak_text.strip():
                            print("  - Found vedtak text on main page")
                        else:
                            print("  - Vedtak text not found on main page, checking subpages...")

                            for i in range(2, 12): 
                                subpage_url = full_url.split('/?lvl=')[0].rstrip('/') + f'/{i}/'
                                print(f"    - Trying subpage: {subpage_url}")

                                try:
                                    sub_response = requests.get(subpage_url)
                                    sub_soup = BeautifulSoup(sub_response.text, 'html.parser')
                                    heading = sub_soup.find('p', string='vedtak:')

                                    if heading:
                                        print("Found 'vedtak:' heading on subpage")
                                        for sibling in heading.find_next_siblings(['p', 'ul']):
                                            vedtak_text+= sibling.text.replace('\n',' ').replace('\r',' ').replace('  ',' ')+' '
                                        print(vedtak_text)
                                        break
                                except Exception as e:
                                    print(f"    - Failed to fetch or parse {subpage_url}: {e}")

                            if not vedtak_text.strip():
                                print("  - Vedtak text not found on any subpage")


                    except:
                        print('Did not find accurate block')
                else:
                    print(f"  - No summary found for case number {sakid}")

            except requests.exceptions.RequestException as e:
                print(f"  - Could not retrieve summary from {full_url}. Error: {e}")

            time.sleep(2)
            break  # Take only the first relevant publication

    saks_data.append({
        'url':full_url,
        'sakid': sakid,
        'tittel': tittel,
        'emneord': emne_liste,
        'sammendrag': sammendrag_tekst,
        'tilrading': vedtak_text
    })

    time.sleep(1.5)


# Step 3: Put data in DataFrame
df_combined = pd.DataFrame(saks_data)
print(df_combined.head())

print(f"\nDataFrame info:")
print(f"Number of rows: {len(df_combined)}")
print(f"Number of columns: {len(df_combined.columns)}")
print(f"Columns: {list(df_combined.columns)}")

sammendrag_count = df_combined['sammendrag'].notna().sum()
print(f"Antall saker med sammendrag: {sammendrag_count}/{len(df_combined)}")

df_combined.to_csv('data2.csv', index=False)