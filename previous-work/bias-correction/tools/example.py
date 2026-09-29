from fuxi_loader import load_fuxi
from rainfall_loader import load_imd, load_imerg


forecast = load_fuxi(date="2002-06-20")
imerg = load_imerg("2002-06-20", "2002-07-31")
imd = load_imd("2002-06-20", "2002-07-31")

print(forecast)
print(imerg)
print(imd)
