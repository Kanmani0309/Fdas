import datetime



# datetime object
dt = datetime.datetime.now()
# printing in dd/mm/YY H:M:S format using strftime()
hours_ = dt.strftime("%H")
hours_= int(hours_)
minutes_= dt.strftime('%M')
minutes_= 1+int(minutes_)
seconds_= dt.strftime('%S')
seconds_= 15+int(seconds_)
if minutes_ == 58 or minutes_ == 59:
    hours_ += 1
if seconds_ >= 45 or seconds_ <= 60 :
    minutes_ += 1

