import os 
import json
# create file to store names and gender 
Name_gender_file= "name_gender.json"

# create function for load genders
def load_gender(Name_gender_file):
    if os.path.exists(Name_gender_file):
        with open (Name_gender_file,"r")as file:
            return json.load(file)
    return{}
# create function to save genders 
def save_gender(data):
    history = load_gender(Name_gender_file)
    history.update(data)
    with open (Name_gender_file,"w")as file:
        json.dump(history,file)
# create predict function
def predict_gender(name):
    name=name.lower()
    gender_data = load_gender(Name_gender_file)
    if name in gender_data:
        return gender_data[name]
    #return "female" if name[-1] in 'aeiu' elif "male" else 
    else:
        if name[-1] in 'aeiy':
            return"Female"
        elif name[-1] in 'nrdohmsv':
            return "Male"
        elif name[-2:] in ['ie']:
            return"Female"
        else:
            return"Neutral"
        
def  update_gender_data(name, gender):
    name = name.lower()
    gender_data = load_gender(Name_gender_file)
    gender_data[name]= gender
    save_gender(gender_data)
    print(f"updated data:{name} is stored as {gender}")

def main():
    while True:
        choice = input("Do you want to predict or train? (p/t): ").lower()
        if choice == 't':
            name=input("Enter name:").strip()
            gender=input("Enter gender:(m/f)").strip().lower()
            if gender in['m','f']:
                update_gender_data(name,gender)
            else:
                print("Invalid gender.Please enter valid gender 'm'for male or 'f' female.")
        elif choice =='p':
            name=input("Enter name:").strip()     
            prediction=predict_gender(name) 
            print(prediction)
            check_prediction=input("Predicted gender is correct or not?(y/n):").lower()
            # if check_prediction =='y':
            #     save_gender(name,prediction)
            if check_prediction == 'n':
                gender=input("Enter correct gender:")
                update_gender_data(name,gender)



            if "Neutral" in prediction.lower():
                # gender_sugesssion=predict_gender(name)
                # print(gender_sugesssion)  
                correct_gender=input(f"What is the correct gender for{name}?(m/f):").strip().lower()
                if correct_gender in ['m','f']:
                    update_gender_data(name,correct_gender)
                else:
                    print("Invalid input,Please enter 'm' for male or 'f' for female.")
        else:
            print("Invalid choice,Please enter 'p' for predict or 't' for train.") 
        #ask user to continue or exit
        continue_choice=input("Do you want to enter another name(choice) or exit the program?(y/n):").strip().lower()
        if continue_choice !='y':
            break
if __name__ =="__main__":

    main()
            
                    
