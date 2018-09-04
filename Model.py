import numpy as np
import pandas as pd
import pickle

from utils import make_dummy
from utils import test_dummy

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

class Train:
	def __init__(self,data):
		self.data = data

		self.Var = ['주야','요일','사망자수','사상자수','중상자수',
				'경상자수','부상신고자수','발생지시도','발생지시군구','사고유형_대분류',
				'사고유형_중분류','법규위반','도로형태_대분류','도로형태','당사자종별_1당_대분류',
				'당사자종별_2당_대분류']

		dummy_dict = {}

		self.category_key = [0,1,7,8,9,10,11,12,13,14,15]
		for i in self.category_key:
			dummy_dict[self.Var[i]] = data[self.Var[i]].unique()

		with open("./Model/dummy_dict.pickle", "wb") as f: 
			pickle.dump(dummy_dict, f)
		self.dummy_dict =dummy_dict

	def A_train(self):
		A_var = [1,5,6,7,8,9,10,11,12,15]
		A_tar = self.Var[0]

		A_train = self.data[self.Var[5:7]].values
		A_key = [i for i in A_var if i in self.category_key]

		for k in A_key:
			A_train = np.concatenate((A_train, make_dummy(self.data[self.Var[k]],self.dummy_dict[self.Var[k]]))
									,axis=1)

		A_label = np.argmax(make_dummy(self.data[A_tar],self.dummy_dict[A_tar]),1)

		A_xgb = xgb.XGBClassifier()
		A_xgb.fit(X=A_train,
		      y = A_label)

		with open("./Model/A_model.pickle", "wb") as f: 
			pickle.dump(A_xgb, f)

	def B_train(self):
		B_var = [0,2,3,4,5,6,7,8,13,14,15]
		B_tar = self.Var[1]

		B_train = self.data[self.Var[2:7]].values
		B_key = [i for i in B_var if i in self.category_key]

		for k in B_key:
			B_train = np.concatenate((B_train, make_dummy(self.data[self.Var[k]],self.dummy_dict[self.Var[k]]))
									,axis=1)
		B_label = np.argmax(make_dummy(self.data[B_tar],self.dummy_dict[B_tar]),1)

		B_xgb = xgb.XGBClassifier()
		B_xgb.fit(X=B_train,
		      y = B_label)

		with open("./Model/B_model.pickle", "wb") as f: 
			pickle.dump(B_xgb, f)

	def C_train(self):
		C_var = [9, 10, 11, 12]
		C_tar = self.Var[2]
		df = self.data[[self.Var[i] for i in C_var]]
		c_train = self.data[self.Var[14:15]].values
		
		for k in C_var:
			c_train = np.concatenate((c_train, make_dummy(self.data[self.Var[k]], self.dummy_dict[self.Var[k]])), axis=1)
		tmp = pd.DataFrame(c_train)
		del tmp[0]

		clf = RandomForestClassifier(n_estimators=100)
		clf.fit(tmp, self.data[C_tar])

		with open("./Model/C_model.pickle", "wb") as f:
			pickle.dump(clf, f)

	def E_train(self):
		E_var = [1,7,9,10,11,12,14,15]
		E_tar = self.Var[4]
		
		E_train = self.data[self.Var[2:4] + self.Var[5:7]].values
		E_key = [i for i in E_var if i in self.category_key]

		for k in E_key:
			E_train = np.concatenate((E_train, make_dummy(self.data[self.Var[k]],self.dummy_dict[self.Var[k]]))
									,axis=1)
		
		E_xgb = xgb.XGBClassifier()
		E_xgb.fit(X=E_train,
		      y = self.data[E_tar])

		with open("./Model/E_model.pickle", "wb") as f: 
			pickle.dump(E_xgb, f)

	def F_train(self):
		F_var = [9, 10, 11, 12]
		F_tar = self.Var[5]
		df = self.data[[self.Var[i] for i in F_var]]
		
		c_train = self.data[self.Var[14:15]].values
		for k in F_var:
			c_train = np.concatenate((c_train, make_dummy(self.data[self.Var[k]], self.dummy_dict[self.Var[k]])), axis=1)
		tmp = pd.DataFrame(c_train)
		del tmp[0]

		clf = RandomForestClassifier(n_estimators=100)
		clf.fit(tmp, self.data[F_tar])

		with open("./Model/F_model.pickle", "wb") as f:
			pickle.dump(clf, f)

	def G_train(self):
		G_var = [0,1,2,5,7,9,10,11,13,14,15]
		G_tar = self.Var[6]

		G_train = self.data[[self.Var[i] for i in [2,5]]].values
		G_key = [i for i in G_var if i in self.category_key]

		for k in G_key:
			G_train = np.concatenate((G_train, make_dummy(self.data[self.Var[k]],self.dummy_dict[self.Var[k]])),axis=1)

		G_rf = RandomForestClassifier()
		G_rf.fit(G_train, self.data[G_tar])

		with open("./Model/G_model.pickle", "wb") as f: 
			pickle.dump(G_rf, f)


	def HI_train(self):
		df = self.data[[self.Var[i] for i in self.category_key]]
		with open('./Model/HI_data.pickle', 'wb') as f:
			pickle.dump(df, f)

	def K_train(self):
		K_var = [0,2,3,4,5,6,7,8,13,14,15]
		K_tar = self.Var[10]
		
		K_train = self.data[self.Var[2:7]].values
		K_key = [i for i in K_var if i in self.category_key]

		for k in K_key:
			K_train = np.concatenate((K_train, make_dummy(self.data[self.Var[k]],self.dummy_dict[self.Var[k]]))
									,axis=1)
		
		K_label = np.argmax(make_dummy(self.data[K_tar],self.dummy_dict[K_tar]),1)

		K_xgb = xgb.XGBClassifier()
		K_xgb.fit(X=K_train,
		      y = K_label)

		with open("./Model/K_model.pickle", "wb") as f: 
			pickle.dump(K_xgb, f)


	def MN_train(self):
		self.data['count']=1
		part_data = self.data[['사고유형_중분류','법규위반','도로형태_대분류','도로형태','count']]
		rule_data = pd.DataFrame(part_data.groupby(['사고유형_중분류','법규위반','도로형태_대분류','도로형태']).count().reset_index())
		rule_data.to_pickle("./Model/MN_model.pkl")
		

	def OP_train(self):
		OP_var = [1,4,5,6,7,8,9,10,11]
		O_tar = self.Var[14]
		P_tar = self.Var[15]

		OP_train = self.data[self.Var[4:7]].values
		OP_key = [i for i in OP_var if i in self.category_key]

		for k in OP_key:
			OP_train = np.concatenate((OP_train, make_dummy(self.data[self.Var[k]],self.dummy_dict[self.Var[k]]))
									,axis=1)
		O_label = np.argmax(make_dummy(self.data[O_tar],self.dummy_dict[O_tar]),1)
		P_label = np.argmax(make_dummy(self.data[P_tar],self.dummy_dict[P_tar]),1)

		O_label = make_dummy(self.data[O_tar],self.dummy_dict[O_tar])
		P_label = make_dummy(self.data[P_tar],self.dummy_dict[P_tar])

		O_rf = RandomForestClassifier(n_estimators=100)
		O_rf.fit(X=OP_train,y=O_label)
		P_rf = RandomForestClassifier(n_estimators=100)
		P_rf.fit(X=OP_train,y=P_label)
		
		with open("./Model/O_model.pickle", "wb") as f: 
			pickle.dump(O_rf, f)
		with open("./Model/P_model.pickle", "wb") as f: 
			pickle.dump(P_rf, f)

	def train(self):
		self.A_train()
		self.B_train()
		self.C_train()
		self.E_train()
		self.F_train()
		self.G_train()
		self.HI_train()
		self.K_train()
		self.MN_train()
		self.OP_train()
		print('Train Complete!')




class Predict:
	def __init__(self,row,col,data):
		self.row = row
		self.col = col
		self.data = data
		self.dummy_dict = pickle.load(open("./Model/dummy_dict.pickle", "rb"))
		self.Var =['주야','요일','사망자수','사상자수','중상자수',
					'경상자수','부상신고자수','발생지시도','발생지시군구','사고유형_대분류',
					'사고유형_중분류','법규위반','도로형태_대분류','도로형태','당사자종별_1당_대분류',
					'당사자종별_2당_대분류']
		
		self.category_key = [0,1,7,8,9,10,11,12,13,14,15]

	def A_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		var = [1,5,6,7,8,9,10,11,12,15]
		tar = Var[0]

		test = list(ex[Var[5:7]].values)
		key = [i for i in var if i in category_key]
		for k in key:
			test.extend(test_dummy(ex[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1,-1)

		model = pickle.load(open("./Model/A_model.pickle", "rb"))
		pred = model.predict(test)

		return dummy_dict[tar][pred][0]

	def B_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		var = [0,2,3,4,5,6,7,8,13,14,15]
		tar = Var[1]

		test = list(ex[Var[2:7]].values)
		key = [i for i in var if i in category_key]
		for k in key:
			test.extend(test_dummy(ex[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1,-1)

		model = pickle.load(open("./Model/B_model.pickle", "rb"))
		pred = model.predict(test)

		return dummy_dict[tar][pred][0]

	def C_model(self,row,data,dummy_dict,Var, category_key):
		df = data.iloc[row]
		C_var = [9, 10, 11, 12]
		C_tar = Var[2]
		df = df[[Var[i] for i in C_var]]
		test = []

		for k in C_var:
			test.extend(test_dummy(df[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1, -1)

		model = pickle.load(open("./Model/C_model.pickle", "rb"))

		return model.predict(test)[0]
	
	def D_model(self,row,data,dummy_dict,Var,category_key):
	    df = data.iloc[row]
	    
	    model =[self.C_model, self.E_model, self.F_model, self.G_model]
	    ans = 0
	    for i in range(4):
	        if pd.isnull(i):
	        	ans += model[idx](row, data,dummy_dict, Var, category_key)
	    return ans
    
	def E_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		var = [1,7,9,10,11,12,14,15]
		tar = Var[4]

		test = list(ex[Var[2:4] + Var[5:7]].values)
		key = [i for i in var if i in category_key]
		for k in key:
			test.extend(test_dummy(ex[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1,-1)

		model = pickle.load(open("./Model/E_model.pickle", "rb"))
		pred = model.predict(data=test)

		return pred[0]

	def F_model(self,row,data,dummy_dict,Var, category_key):
		df = data.iloc[row]
		C_var = [9, 10, 11, 12]
		C_tar = Var[5]
		df = df[[Var[i] for i in C_var]]
		test = []

		for k in C_var:
			test.extend(test_dummy(df[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1, -1)

		model = pickle.load(open("./Model/F_model.pickle", "rb"))
		return model.predict(test)[0]
	
	def G_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		var = [0,1,2,5,7,9,10,11,13,14,15]
		tar = Var[6]

		test = list(ex[[Var[i] for i in [2,5]]].values)
		key = [i for i in var if i in category_key]

		for k in key:
			test.extend(test_dummy(ex[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1,-1)

		model = pickle.load(open("./Model/G_model.pickle", "rb"))
		return model.predict(test)[0]


	def H_model(self,row,data,dummy_dict,Var,category_key):
		df = pickle.load(open("./Model/HI_data.pickle", "rb"))
		
		sample = df.loc[np.where((df["사고유형_대분류"] == data["사고유형_대분류"].loc[row])
								  & (df["사고유형_중분류"] == data["사고유형_중분류"].loc[row])
								  & (df["법규위반"] == data["법규위반"].loc[row]))]
		
		if len(sample) == 0:
			return df["발생지시도"].describe().top
		else:
			return sample["발생지시도"].describe().top
	
	def I_model(self,row,data,dummy_dict,Var,category_key):
		df = pickle.load(open("./Model/HI_data.pickle", "rb"))

		if pd.isnull(data["발생지시도"].loc[row]):
			sample = df.loc[np.where((df["발생지시도"] == self.H_model(row,data,dummy_dict,Var,category_key))
									  & (df["사고유형_대분류"] == data["사고유형_대분류"].loc[row])
									  & (df["사고유형_중분류"] == data["사고유형_중분류"].loc[row])
									  & (df["법규위반"] == data["법규위반"].loc[row]))]
			if len(sample) == 0:
				return df[df["발생지시도"] == data["발생지시도"].loc[row]]["발생지시군구"].describe().top
			else:
				return sample["발생지시군구"].describe().top
		else:
			sample = df.loc[np.where((df["발생지시도"] == data["발생지시도"].loc[row])
									  & (df["사고유형_대분류"] == data["사고유형_대분류"].loc[row])
									  & (df["사고유형_중분류"] == data["사고유형_중분류"].loc[row])
									  & (df["법규위반"] == data["법규위반"].loc[row]))]
			if len(sample) == 0:
				return df[df["발생지시도"] == data["발생지시도"].loc[row]]["발생지시군구"].describe().top
			else:
				return sample["발생지시군구"].describe().top
	
	def J_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		if ex['당사자종별_2당_대분류'] == '없음':
			return '차량단독'
		elif ex['당사자종별_2당_대분류']  == '보행자':
			return '차대사람'
		elif ex['당사자종별_2당_대분류'] == '건널목':
	 		return '건널목'
		else:
	 		return '차대차'

	def K_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		var = [0,2,3,4,5,6,7,8,13,14,15]
		tar = Var[10]

		test = list(ex[Var[2:7]].values)
		key = [i for i in var if i in category_key]
		for k in key:
			test.extend(test_dummy(ex[Var[k]], dummy_dict[Var[k]]))
		
		test = np.array(test).reshape(1,-1)
		model = pickle.load(open("./Model/K_model.pickle", "rb"))
		pred = model.predict(test)

		return dummy_dict[tar][pred][0]

	def L_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]
		rule = pd.read_pickle("./Model/MN_model.pkl")

		var1 = ex['사고유형_중분류']        
		if pd.isnull(var1) == False :
			rule = rule[rule['사고유형_중분류']==var1]

		var3 = ex['도로형태_대분류']
		if pd.isnull(var3) == False :  
			rule = rule[rule['도로형태_대분류']==var3]

		var2 = ex['도로형태']
		if pd.isnull(var2) == False :
			rule = rule[rule['도로형태']==var2]     
		
		if rule.empty == True:  ## 셋다 null이면 단일로로 채우기
			return '안전운전 의무 불이행'
			
		return rule['법규위반'].describe().top
	def M_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]
		rule = pd.read_pickle("./Model/MN_model.pkl")
		
		var3 = ex['도로형태']
		if pd.isnull(var3) == False :  
			rule = rule[rule['도로형태']==var3]

		var1 = ex['사고유형_중분류']        
		if pd.isnull(var1) == False :
			rule = rule[rule['사고유형_중분류']==var1]

		var2 = ex['법규위반']
		if pd.isnull(var2) == False :
			rule = rule[rule['법규위반']==var2]     
		
		if rule.empty == True:  ## 셋다 null이면 단일로로 채우기
			return '단일로'
		
		return rule['도로형태_대분류'].describe().top
	def N_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]
		rule = pd.read_pickle("./Model/MN_model.pkl")

		var3 = ex['도로형태_대분류'] 
		if pd.isnull(var3) == False :  
			rule = rule[rule['도로형태_대분류']==var3]

		if pd.isnull(var3) == True :
			var3 = self.M_model(row,data,dummy_dict,Var,category_key)
			rule = rule[rule['도로형태_대분류']==var3]

		var1 = ex['사고유형_중분류']        
		if pd.isnull(var1) == False :
			rule = rule[rule['사고유형_중분류']==var1]
		
		var2 = ex['법규위반']
		if pd.isnull(var2) == False :
			rule = rule[rule['법규위반']==var2]     

		if rule.empty ==True: 
			return '기타단일로'

		return rule['도로형태'].describe().top
	def O_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		var = [1,4,5,6,7,8,9,10,11]
		tar = Var[14]

		test = list(ex[Var[4:7]].values)
		key = [i for i in var if i in category_key]
		
		for k in key:
			test.extend(test_dummy(ex[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1,-1)

		model = pickle.load(open("./Model/O_model.pickle", "rb"))
		pred = np.argmax(model.predict(X=test),1)

		return dummy_dict[tar][pred][0]
	
	def P_model(self,row,data,dummy_dict,Var,category_key):
		ex = data.iloc[row]

		var = [1,4,5,6,7,8,9,10,11]
		tar = Var[15]
		
		test = list(ex[Var[4:7]].values)
		key = [i for i in var if i in category_key]
		
		for k in key:
			test.extend(test_dummy(ex[Var[k]], dummy_dict[Var[k]]))
		test = np.array(test).reshape(1,-1)

		model = pickle.load(open("./Model/P_model.pickle", "rb"))
		pred = np.argmax(model.predict(X=test),1)

		return dummy_dict[tar][pred][0]

	def predict(self):
		self.model = [self.A_model,self.B_model,self.C_model,self.D_model,self.E_model
				,self.F_model,self.G_model,self.H_model,self.I_model,self.J_model
				,self.K_model,self.L_model,self.M_model,self.N_model,self.O_model
				,self.P_model]
		result = []
		for i,j in zip(self.row,self.col):
			print(i,j)
			result.append(self.model[ord(j)-65](i,self.data,self.dummy_dict,self.Var,self.category_key))
		return result



