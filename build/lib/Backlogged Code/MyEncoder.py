# Extended subclass of LabelEncoder
class MyEncoder(preprocessing.LabelEncoder):
    
    def __init__(self,input_labels):
        
        # Error check parameter variable
        if not isinstance(input_labels, (list, tuple, np.ndarray)):
            exit("You sent a object that is not an array type!")
        
        # Create label encoder and fit the labels
        self.fit(input_labels)
        
        # Creating dict
        self.__encode_label = defaultdict(dict)
        self.__encode_num = defaultdict(dict)
        
        # Init dict
        for i, item in enumerate(self.classes_):
            self.__encode_label[item] = i
            self.__encode_num[i] = item

    # Print the label to number relationship
    def displayEncodedMap(self):
        
        print("\nLabel mapping:\n")
        for i, item in enumerate(self.__encoder.classes_):
            print("\t",item, '-->', i)
            
            # Draw a line
            print("-"*30,"\n")
        
        # Decode the label to the num associated with it
        def decodeLabel(self,labels):
            if (not isinstance(labels,str)) and ( not isinstance(labels, (list, tuple, np.ndarray))):
                exit("You sent value(s) that are not label(s)!")
        
            if isinstance(labels,str):
                
                if labels in self.__encode_label:
                    return self.__encode_label[labels]
                else:
                    exit("This key does not exist in the encoder!")

            else:
                return list(self.transform(labels))
        
        # Decode the num to the label associated with it
        def decodeNum(self,values):
            if (not isinstance(values,int)) and (not isinstance(values, (list, tuple, np.ndarray))):
                exit("You sent value(s) that are not integer(s)!")
        
            if isinstance(values,int):
                
                if values in self.__encode_num:
                    return self.__encode_num[values]
                else:
                    exit("This key does not exist in the encoder!")

            else:
                return list(self.inverse_transform(values))
