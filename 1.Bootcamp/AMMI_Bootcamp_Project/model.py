
class RuleBasedModel():

    def predict(self, sample):
        if (sample['f4']+sample['f8']) < 0.45:
          return 2
        else:
          return 1

    def accuracy(self, data):
        predictions = []
        targets = []
        for i in data:
            predictions.append(self.predict(i))
            targets.append(int(i['class']))
        counter=0
        for prediction,target in zip(predictions,targets):
            if prediction==target:
                counter += 1
        accuracy = counter/len(targets)
        return accuracy*100 ,counter,len(data)

    def predict_all(self, data):
        predictions = []
        for i in data:
            predictions.append(self.predict(i))
        return predictions

    def predict_byID(self,data,ID):
        for i in data:
            if i['ID']==int(ID):
                return self.predict(i)
