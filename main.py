import os

print("Select which action do you want to make: ")
print("     1. Train the model.")
print("     2. Execute the trained model.")
sel = input("Your Selection: ")
sel = int(sel)

if sel == 1:
    os.system("python tweets_classifier.py")
elif sel == 2:
    os.system("python m_learning_tweets.py &")
    os.system("python visualization_app.py")
else:
    print("Do not know input.")