# classification => 分類
# regression => 回帰

class QuestionType:
    @classmethod
    def setting(cls, ques):
        if(ques in [ 'class', 'classification']):
            return {'out_func': 'softmax', 'loss': 'cross'}
        elif(ques in ['reg', 'regres', 'regression']):
            return {'out_func': 'id', 'loss': 'square'}
        return {'out_func': None, 'loss': None}

