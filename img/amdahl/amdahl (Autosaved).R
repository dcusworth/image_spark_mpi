dat = read.table("~/Box Sync/School/2017 Spring/CS 205/image_spark_mpi/img/amdahl/opemp_result.csv", header=T, sep=",")
colnames(dat) = c("threads", "parallel time", "overhead time", "total time")

dat2 = t(as.matrix(dat[,c(4,2,3)]))
colnames(dat2) = dat[,1]

barplot(dat2, beside=T, xlab="threads", ylab="time (s)", legend=colnames(dat)[c(4,2,3)], border=NA, col=c("black","firebrick4","indianred2"))


dat1 = read.table("~/Box Sync/School/2017 Spring/CS 205/image_spark_mpi/img/amdahl/serial_scale.csv", header=T, sep=",")

df1 = data.frame(x=dat1$N, y=dat1$time)
lm1 = lm(y~x, data=df1)
df2 = data.frame(x=dat1$N, y=dat1$time[1] * (dat1$N/dat1$N[1]))
lm2 = lm(y~x, data=df2)

plot(dat1$N, dat1$time, type="o", lwd=5, ylim=c(0,500), col="dodgerblue", ylab="time (s)", xlab="number of trained images")
lines(dat1$N, dat1$time[1] * (dat1$N/dat1$N[1]), lwd=5, col="slategrey", type="o")
title("Serial code - Model performance vs. # trained images - full comp")
legend("topleft", c("Scaling exactly with size", "Actual result"), bty="n", lty=1, lwd=3, col=c("slategrey","dodgerblue"))