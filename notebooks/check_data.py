import pandas as pd

df = pd.read_csv('../data/master_dataset.csv')

print(f'Load mean:        {df["load_kw"].mean():.1f} kW')
print(f'Load max:         {df["load_kw"].max():.1f} kW')
print(f'Load min:         {df["load_kw"].min():.1f} kW')
print(f'Load total:       {df["load_kw"].sum():.0f} kWh')
print()
print(f'Solar mean(x0.75): {(df["ssrd_wm2"]*0.75).mean():.1f} kW')
print(f'Solar max (x0.75): {(df["ssrd_wm2"]*0.75).max():.1f} kW')
print()
print(f'Solar/Load ratio:  {(df["ssrd_wm2"]*0.75).mean()/df["load_kw"].mean():.2f}')
print(f'Hours solar>load:  {((df["ssrd_wm2"]*0.75) > df["load_kw"]).mean()*100:.1f}%')
print()
print('Per location:')
for loc in ['Tamale','Kumasi','Axim']:
    d = df[df['location']==loc]
    print(f'  {loc}: load_mean={d["load_kw"].mean():.1f} kW  solar_mean={d["ssrd_wm2"].mean()*0.75:.1f} kW')
