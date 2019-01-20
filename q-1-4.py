import numpy as np
import pandas as pd
import random
import seaborn as sns


def main():
    #read data from csv file
    df = pd.read_csv("data.csv")
    
     #Visualisation
    graph = sns.catplot(y="satisfaction_level", x="left",kind="box", hue="number_project", data=df)
    graph.savefig("P4_SatLvl_Vs_NumProjects.png")
    # graph = sns.catplot(y="satisfaction_level", x="left",kind="box", hue="last_evaluation", data=df)
    # graph.savefig("SatLvl_Vs_LastEval.png")
    # sns.catplot(y="satisfaction_level", x="left", hue="time_spend_company", kind = "box",data=df)

    # sns.catplot(y="time_spend_company", x="left",hue="promotion_last_5years", kind = "box",data=df)
    graph = sns.catplot(y="satisfaction_level", x="left",hue="department", kind = "box",data=df)
    graph.savefig("P4_SatLvl_Vs_Dept.png")
    # sns.catplot(y="average_montly_hours", x="left",hue="promotion_last_5years", kind = "box",data=df)
    # sns.catplot(y="average_montly_hours", x="left",hue="promotion_last_5years", data=df)
    
    # sns.catplot(y="satisfaction_level", x="left",hue="promotion_last_5years", data=df)
    # sns.catplot(y="satisfaction_level", x="left",hue="Work_accident", data=df)
    # sns.catplot(y="satisfaction_level", x="left", hue="time_spend_company", data=df)
    # sns.catplot(y="satisfaction_level", x="left", hue="number_project", data=df.query("number_project != 3"))
    # sns.catplot(y="satisfaction_level", x="left", hue="number_project", data=df)
    # sns.catplot(y="satisfaction_level", x="left",kind="swarm", hue="number_project",data=df)
    

# Note : Handling of blank values in test example remianing
# main segment starts here
if __name__ == '__main__':
    main()
    # test()
