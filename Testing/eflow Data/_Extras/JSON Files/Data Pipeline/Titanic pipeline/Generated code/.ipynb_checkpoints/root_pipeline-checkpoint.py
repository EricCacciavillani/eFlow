# Segment Path ID: _Extras/JSON Files/Data Pipeline Segments/DataTransformer/52EAC97BCB.json
# Segment Name:    Remove Unwanted Features
feature_names = ['Name', 'Ticket', 'PassengerId']
if isinstance(feature_names, str):
    feature_names = [feature_names]

for feature_n in feature_names:
    check_if_feature_exists(df,
                            feature_n)
    df.drop(columns=[feature_n],
            inplace=True)


#------------------------------
