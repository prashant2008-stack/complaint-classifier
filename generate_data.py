"""Sample dataset banata hai (agar organizers ka dataset na mile to). Real CSV ho to isko skip karo."""
import random, pandas as pd
random.seed(42)
DATA = {
 "Billing": (["I was charged twice for my order","The invoice amount is wrong","Extra charges added to my bill","Payment deducted but no receipt","My subscription fee increased without notice","Wrong tax applied on my invoice"],
             ["Please correct the bill.","I want this fixed immediately.","Kindly check my account.","This is the second month."]),
 "Delivery": (["My parcel has not arrived yet","The package is delayed by two weeks","Courier marked delivered but I got nothing","Wrong address delivery happened","Tracking shows no update for days","Delivery boy was rude and left the item outside"],
              ["Need it urgently.","Please track my shipment.","This is very late.","Send it again."]),
 "Product Quality": (["The product arrived damaged","Item stopped working after two days","Material quality is very poor","Received a defective piece","The product looks different from the photos","Screen has scratches out of the box"],
                     ["Very disappointed.","Not worth the price.","Expected much better.","Quality control is bad."]),
 "Technical Support": (["The app keeps crashing on login","I cannot reset my password","Website shows error 500 while paying","OTP is not coming on my phone","Software update broke the features","Unable to connect the device to wifi"],
                       ["Please help me fix this.","Tried reinstalling already.","Need technical assistance.","Issue is still not solved."]),
 "Refund & Return": (["I returned the item but refund is pending","Refund not received after 15 days","Return request was rejected without reason","Want to exchange the product for a different size","Money not credited back to my account","Replacement was promised but never sent"],
                     ["Process my refund now.","Waiting for a long time.","Please update me.","This is unacceptable."]),
 "Customer Service": (["Support agent was very rude to me","No one is answering my calls","Nobody replied to my emails for a week","I was transferred between departments many times","The staff behaviour was unprofessional","Chat support closed my ticket without solving it"],
                      ["I want to speak to a manager.","Poor service overall.","Please improve your team.","Escalate my case."]),
}
rows=[]
for cat,(a,b) in DATA.items():
    for _ in range(100):
        t=random.choice(a)+". "+random.choice(b)
        if random.random()<0.15:  # thoda noise, taaki accuracy 100% na ho
            other=random.choice([c for c in DATA if c!=cat]); t=t+" "+random.choice(DATA[other][0])+"."
        rows.append((t,cat))
random.shuffle(rows)
pd.DataFrame(rows,columns=["complaint","category"]).to_csv("complaints.csv",index=False)
print(len(rows),"rows saved")
