makebatches <- function(){
  train$x <- train$x /255
  totnum <- nrow(train$x)
  set.seed(0)
  randomorder <- sample(totnum)
  numdims <- ncol(train$x)

  numbatches <- totnum / 100
  batchsize <- 100
  batchdata <- array(rep(NaN, 100*numdims*numbatches), c( 100, numdims, numbatches))
  
  for (b in 1:numbatches){
    batchdata[,,b] <- train$x[randomorder[(1+(b-1)*batchsize):(b*batchsize)],]
  }
  batchdata

}

maketestbatches <- function(){
test$x <- test$x /255
totnum <- nrow(test$x)
set.seed(0)
randomorder <- sample(totnum)
numdims <- ncol(test$x)
numbatches <- totnum / 100
batchsize <- 100
testbatchdata <- array(rep(NaN, 100*numdims*numbatches), c( 100, numdims, numbatches))

for (b in 1:numbatches){
  testbatchdata[,,b] <- test$x[randomorder[(1+(b-1)*batchsize):(b*batchsize)],]
}
testbatchdata
}