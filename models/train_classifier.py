import sys
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
import pickle


def load_data(database_filepath):
    """
    Inputs:
    database_filepath: The location of the database containing the table
                        which consist the data.
          
    Outputs:
    X : The feature to be considered as input for the model
    Y : The features considered to be as output
    columns: Name of features of the output
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_table',engine)
    X = df['message']
    Y = df[['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']]
    return X,Y,Y.columns


def tokenize(text):
    """
    Input: 
    text:string containg words from the input message

    Output:
    clean_tokens: array of words in all lowercase, stopwords removed and lemmatized
    """
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w not in stopword.words("english")]
    lemmetized = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatized.strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    """
    Input:
    
    Output:
    pipeline: Machine Learning pipeline containing all the ML models
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
     parameters = {
              'tfidf__use_idf':[True,False],
              'tfidf__smooth_idf':[True,False],
              'tfidf__sublinear_tf':[True,False],
              }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Input:
    model: Machine Learning Pipeline
    X_test: Test input part of the splitted dataset
    Y_test: Test output part of the splitted dataset
    category_name: Output category of the message
    Output:
    """
    y_pred = model.predict(X_test)
    y_preds = pd.DataFrame(y_pred,columns=Y_test.columns)
    for col in Y_test.columns:
        print("Report for column:{}\n".format(col),classification_report(Y_test[col],y_preds[col]))
    pass


def save_model(model, model_filepath):
    """
    Input:
    model: ML pipeline
    model_filepath: trained model file location to be saved
    Output:
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
