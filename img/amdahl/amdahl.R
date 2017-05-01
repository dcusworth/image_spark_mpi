dat = read.table("~/Box Sync/School/2017 Spring/CS 205/image_spark_mpi/img/amdahl/opemp_result.csv", header=T, sep=",")
colnames(dat) = c("threads", "parallel time", "overhead time", "total time")

dat2 = t(as.matrix(dat[,c(4,2,3)]))
colnames(dat2) = dat[,1]

barplot(dat2, beside=T, xlab="threads", ylab="time (s)", legend=colnames(dat)[c(4,2,3)], border=NA, col=c("black","firebrick4","indianred2"))
