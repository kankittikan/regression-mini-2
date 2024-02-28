import pandas as pd

def readDataset():
    storage_options = {'User-Agent': 'Mozilla/5.0'}
    # อ่านข้อมูลจากไฟล์ csv
    df = pd.read_csv('https://content.doksakura.com/ml/datasets/Apple462.csv', storage_options=storage_options)
    # ตัด Column 'Unnamed: 0' ออก
    df = df.drop(['Unnamed: 0'], axis=1)
    # เลือกแสดงเฉพาะข้อมูลที่มี region เป็น China แล้วตัด Column 'region' ออก
    df = df[df['region'] == 'China']
    df = df.drop(['region'], axis=1)
    
    return df

def datasetPrepare(df):
    # เรียงข้อมูลโดยเรียงจากวันที่น้อยไปวันที่มาก
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.set_index('Date')
    df = df.sort_index()
    # ทำการแปลงข้อมูลให้เป็น numeric และแทนที่ค่าที่เป็นช่องว่าง
    df = df.replace(" ", "")
    df[['Fuji', 'Envi', 'Gala']] = df[['Fuji', 'Envi', 'Gala']].apply(pd.to_numeric)
    df = df.interpolate()

    return df

def getGalaDatasets():
    df = readDataset()
    df = datasetPrepare(df)
    return df['Gala']

def getEnviDatasets():
    df = readDataset()
    df = datasetPrepare(df)
    return df['Envi']

def getFujiDatasets():
    df = readDataset()
    df = datasetPrepare(df)
    return df['Fuji']