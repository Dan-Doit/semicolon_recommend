from graphqlclient import GraphQLClient
import json
import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#
# key = os.environ['PRISMA_ENDPOINT']

client = GraphQLClient("END PAGE")

arr = client.execute('''

{
users{
  id
  likes{
    post{
      id
    }
  }
}
}

''',
                     )
arr = json.loads(arr)

parr = client.execute('''
{
posts{
  id
}
}
''')
parr = json.loads(parr)

arr = arr['data']['users']

userId = []
for i in arr:
    userId.append(i['id'])

newArr = []

for i in arr:

    for j in i['likes']:
        newArr.append([i['id'], j['post']['id']])

users = []
posts = []
isLike = []
result = []
for i in arr:
    users.append(i['id'])
# 모든 유저값

for i in parr['data']['posts']:
    posts.append(i['id'])
# 모든 포스트값

# 좋아하는지 체크
for i in arr:
    for j in i['likes']:
        isLike.append([i['id'], j['post']['id']])

for i in range(len(users)):
    for j in range(len(posts)):
        for z in isLike:
            if z[0] == users[i] and z[1] == posts[j]:
                count = 5
                break
            else:
                count = 0
        result.append([users[i], posts[j], count])

people = len(users)

users = []
posts = []
isLike = []
temp = []

# 여기서 데이터 분류
for i in range(len(result)):
    users.append(result[i][0])
    posts.append(result[i][1])
    isLike.append(result[i][2])
    temp.append(count)
    count = count + 1
ratings_df = pd.DataFrame({'userId': users, 'postId': posts, 'rating': isLike, 'count': temp})

train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=1234)

"""### Sparse Matrix 만들기"""

sparse_matrix = train_df.groupby('postId').apply(lambda x: pd.Series(x['rating'].values, index=x['userId'])).unstack()
sparse_matrix.index.name = 'postId'

sparse_matrix

# fill sparse matrix with average of post ratings
sparse_matrix_withpost = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)

# fill sparse matrix with average of user ratings
sparse_matrix_withuser = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)

sparse_matrix_withpost

sparse_matrix_withuser

"""## Matrix Factorization with SVD"""

def get_svd(s_matrix, k=people):
    u, s, vh = np.linalg.svd(s_matrix.transpose())
    S = s[:k] * np.identity(k, np.float)
    T = u[:, :k]
    Dt = vh[:k, :]

    item_factors = np.transpose(np.matmul(S, Dt))
    user_factors = np.transpose(T)

    return item_factors, user_factors


item_factors, user_factors = get_svd(sparse_matrix_withpost)
prediction_result_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                    columns=sparse_matrix_withpost.columns.values,
                                    index=sparse_matrix_withpost.index.values)

recommend = prediction_result_df.transpose()



favorite = {}
postId = []

for i in users:
    favorite.setdefault(str(i),[])

for i in range(len(recommend.index)):
  for j in range(len(recommend.columns)):
    if recommend.values[i][j] >= 0.1 and recommend.values[i][j] < 4.9:
      favorite[str(recommend.index[i])].append(recommend.columns[j])
    print(recommend.index[i], recommend.columns[j], round(recommend.values[i][j]))

from flask import Flask
from flask_restful import Resource, Api
from flask import jsonify


app = Flask(__name__)
api = Api(app)


class RegistUser(Resource):
    def get(self):
        return jsonify(favorite)


api.add_resource(RegistUser, '/recommendation/key=teamsemicolon')

if __name__ == '__main__':
    app.run(host="172.30.1.23",port=5000, debug=True)

