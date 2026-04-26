import pandas as pd
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Corpus sourced from: Ahmed et al. (2026), "BanglaRegionalTextCorpus: A Curated Dataset for
# Four Regional Bangla Dialects", Data in Brief, https://doi.org/10.1016/j.dib.2026.001381
df=pd.read_excel("BanglaRegionalTextCorpus-tnQRMn.xlsx")

def romanize(text):
    if not isinstance(text,str): return ""
    return transliterate(text,sanscript.BENGALI,sanscript.ITRANS)

print("Romanizing...")
regional_rom = df["Regional_Texts"].apply(romanize)
standard_rom = df["Standard_Bangla_Texts"].apply(romanize)

with open("corpus_aligned.txt","w",encoding="utf-8") as f:
    for r,s in zip(regional_rom,standard_rom):
        if r and s:
            f.write(f"{r}|||{s}\n")

print(f"Done. {len(df)} pairs -> corpus_aligned.txt")
print("\nSample:")
for r,s in zip(regional_rom[:3],standard_rom[:3]):
    print(f"  Regional : {r}")
    print(f"  Standard : {s}")
    print()
