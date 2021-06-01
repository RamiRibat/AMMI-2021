from utils import reader
from model import RuleBasedModel


def main(arg):

    train_file = 'Data/train_data.txt'
    test_file = 'Data/test_data.txt'
    variables = ['ID', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'class']
    #
    #
    print("========= Reading train dataset =========")
    # TO DO:
    # use the read data function you created to read the train data
    train_data = reader(train_file, variables)
    print("======== Done reading =========.\n")


    print("========= Reading test data =========")
    # TO-DO
    # Read the test  data
    test_data = reader(test_file, variables)
    print("========= Done reading =========.\n")




    print("==== Training classifier =====")
    # TO-DO
    # Initialize the classifier you built in model.py and return the necessary values
    model = RuleBasedModel()
    print("======== Done training classifier ===========.\n")


    print("========= Classifying test samples =======")
    # TO-DO
    # use your classifier to do predictions on all the test samples
    predictions = model.predict_all(test_data)
    for i in range(len(predictions)):
            test_data[i]['prediction'] = predictions[i]
    print("========== Done classifying =======")

    # TO-DO
    # Evalutate your classifier with the Accuracy function you implemented and return the necessary outputs
    accuracy,numCorrect,total_samples = model.accuracy(test_data)
    print(f"Model's Accuracy {round(accuracy)} %, model correctly predicted {numCorrect} out of {total_samples}")
    print('================================================================')

    if arg:
        print("========== Prediction ==========")
        print(f"Prediction for ID={arg}: { model.predict_byID(test_data, arg[0]) }")
        print("======== Done Prediction =======")

    print("finished.\n")

# main()


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Predict using an ID.')
    parser.add_argument('--predict', type=int, help='Provide an ID to be predicted.')
    args = parser.parse_args()
    print(args.predict)
    main(args.predict)
