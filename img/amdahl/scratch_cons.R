hrs = 0:23
dates = seq(as.Date("2015-10-01"), as.Date("2015-11-30"), by=1)
df = data.frame(DATES = rep(dates, each=length(hrs)), HH = rep(hrs, times=length(dates)))
write.csv(df, "~/Desktop/consulate2015.csv")


#2015
con2015 = read.table("~/Desktop/consulate2015b.csv", header=T, sep=",")
c2015 = data.frame(DATES = as.character(as.Date(con2015[,"DATES"], "%m/%d/%y")), HH=as.numeric(con2015[,"HH"]), PM2.5 = as.numeric(as.character(con2015[,"PM2.5"])))
c2015[c2015[,"PM2.5"] == -999, "PM2.5"] = NA

write.csv(c2015, "~/Desktop/consulate2015.csv", row.names=F)


#2014
con2014 = read.table("~/Desktop/consulate2014b.csv", header=F)
colnames(con2014) = c("date", "time", "AMPM", "delhipm", "delhiaqi", "chennaipm", "aqichenn", "pmkol")
con2014[con2014 == "NoData"] = -999

ctime = as.character(con2014[,"time"])
cdate = as.Date(as.character(con2014[,"date"]), "%d/%m/%y")
cdate[cdate > as.Date("2020-01-01")] = as.Date(as.character(con2014[,"date"]), "%d/%m/%Y")[cdate > as.Date("2020-01-01")]

c2014 = data.frame(DATES = as.character(cdate), HH = as.numeric(substr(ctime, 1, nchar(ctime)-3)), PM2.5 = as.numeric(as.character(con2014[,"delhipm"])))
c2014[c2014[,"PM2.5"] == -999, "PM2.5"] = NA


write.csv(c2014, "~/Desktop/consulate2014.csv", row.names=F)


#2013
con2013 = read.table("~/Desktop/consulate2013b.csv", header=F)
colnames(con2013) = c("date", "time", "AMPM", "delhipm", "delhiaqi", "chennaipm", "aqichenn", "pmkol")
con2013[con2013 == "NoData"] = -999

ctime = as.character(con2013[,"time"])
cdate = as.Date(as.character(con2013[,"date"]), "%d/%m/%y")
cdate[cdate > as.Date("2020-01-01")] = as.Date(as.character(con2013[,"date"]), "%d/%m/%Y")[cdate > as.Date("2020-01-01")]

c2013 = data.frame(DATES = as.character(cdate), HH = as.numeric(substr(ctime, 1, nchar(ctime)-3)), PM2.5 = as.numeric(as.character(con2013[,"delhipm"])))
c2013[c2013[,"PM2.5"] == -999, "PM2.5"] = NA

write.csv(c2013, "~/Desktop/consulate2013.csv", row.names=F)

