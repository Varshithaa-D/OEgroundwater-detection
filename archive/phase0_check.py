import pandas as pd

gee_taluks = [
  'Bagalkote', 'Bailhongal', 'Savadatti', 'Devanahalli', 'Doddaballapura', 'Hoskote', 'Nelamangala',
  'Anekal', 'Bangalore-East', 'Bangalore-North', 'Bangalore-South', 'Yelahanka',
  'Harohalli', 'Magadi', 'Channapatna', 'Kanakpura', 'Gundlupet',
  'Bagepalli', 'Chikkaballapura', 'Chinthamani', 'Gauribidanur', 'Gudibande', 'Shidlagatta',
  'Ajjampura', 'Challakere', 'Chitradurga', 'Hiriyur', 'Holalkere', 'Hosadurga', 'Molakalmuru',
  'Channagiri', 'Jagaluru', 'Harihar', 'Honnali', 'Gajendragad', 'Naragund',
  'Arsikere', 'Channarayapatna', 'Arkalgud', 'Bangarpet', 'Kolar Gold Field', 'Kolara', 'Malur', 
  'Mulabagilu', 'Srinivaspura', 'Chiknayakanahalli', 'Madhugiri', 'Sira', 'Tiptur', 'Tumakuru', 
  'Pavagada', 'Kunigal', 'Hagaribommanahalli', 'Kotturu'
]

oe_df = pd.read_csv(r"manual_dataset&ground_truth\gwra_2025_oe_taluks.csv")

# Clean strings
oe_df['taluk_clean'] = oe_df['taluk'].str.lower().str.strip()
gee_clean = [t.lower().strip() for t in gee_taluks]

filtered = oe_df[oe_df['taluk_clean'].isin(gee_clean)]

print(f"Out of 54 GEE Taluks, we found {len(filtered)} in the CSV.")
print("\nBreakdown by Category in CSV:")
print(filtered['category'].value_counts())
print("\nBreakdown by label_binary (1=OE, 0=Safe):")
print(filtered['label_binary'].value_counts())
