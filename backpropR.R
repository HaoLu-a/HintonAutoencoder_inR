maxepoch <- 200 # use 200 in formal, use 1 whend debugging

setwd("H:/HintonAutoencoder/Rcode")

load("mnistvh.RData")
load("mnisthp.RData")
load("mnisthp2.RData")
load("mnistpo.RData")
source("converterR.r")
source("cg_minstR.R")
source("minimize.R")
source("makebatchesR.r")
load_mnist()
batchdata <- makebatches()
testbatchdata <- maketestbatches()

numcases <- dim(batchdata)[1]
numdims <- dim(batchdata)[2]
numbatches <- dim(batchdata)[3]

N <- numcases

# preinitialize weights of the autoencoder
w1 <- rbind(vishid, hidrecbiases)
w2 <- rbind(hidpen, penrecbiases)
w3 <- rbind(hidpen2, penrecbiases2)
w4 <- rbind(hidtop, toprecbiases)
w5 <- rbind(t(hidtop), topgenbiases)
w6 <- rbind(t(hidpen2), hidgenbiases2)
w7 <- rbind(t(hidpen), hidgenbiases)
w8 <- rbind(t(vishid), visbiases)
# end of preinitialization of weights

l1 <- dim(w1)[1] - 1
l2 <- dim(w2)[1] - 1
l3 <- dim(w3)[1] - 1
l4 <- dim(w4)[1] - 1
l5 <- dim(w5)[1] - 1
l6 <- dim(w6)[1] - 1
l7 <- dim(w7)[1] - 1
l8 <- dim(w8)[1] - 1
l9 <- l1

# test_err <- [], train_err<-[]
train_err <- rep(0,maxepoch)
test_err <- rep(0,maxepoch)


for (epoch in 1:maxepoch){
  # Compute training recconstruction error
  err <- 0
  numcases <- dim(batchdata)[1]
  numdims <- dim(batchdata)[2]
  numbatches <- dim(batchdata)[3]
  
  N <- numcases
  
  print("calculating train error")
  for (batch in 1:numbatches){
    print(batch)
    data <- batchdata[,,batch]
    data <- cbind(data,rep(1,N))
    w1probs <- 1./(1 + exp(-data %*% w1))
    w1probs <- cbind(w1probs,rep(1,N))
    w2probs <- 1./(1 + exp(-w1probs %*% w2))
    w2probs <- cbind(w2probs,rep(1,N))
    w3probs <- 1./(1 + exp(-w2probs %*% w3))
    w3probs <- cbind(w3probs,rep(1,N))
    w4probs <- w3probs %*% w4
    w4probs <- cbind(w4probs,rep(1,N))
    w5probs <- 1./(1 + exp(-w4probs %*% w5))
    w5probs <- cbind(w5probs,rep(1,N))
    w6probs <- 1./(1 + exp(-w5probs %*% w6))
    w6probs <- cbind(w6probs,rep(1,N))
    w7probs <- 1./(1 + exp(-w6probs %*% w7))
    w7probs <- cbind(w7probs,rep(1,N))
    dataout <- 1./(1 + exp(-w7probs %*% w8))
    err <- err + 1/N * sum((data[,1:(dim(data)[2]-1)] - dataout)^2)
  }
  train_err[epoch] <- err/numbatches
  
  # end of computing training reconstruction error
  
  # display figure top row real data bottom row reconstructions
  #SKIPPED
  
  # compute test reconstruction error
  
  testnumcases <- dim(testbatchdata)[1]
  testnumdims <- dim(testbatchdata)[2]
  testnumbatches <- dim(testbatchdata)[3]
  
  N <- testnumcases
  
  err <- 0
  print("calculating Test error")
  for (batch in 1:testnumbatches){
    print(batch)
    data <- testbatchdata[,,batch]
    data <- cbind(data,rep(1,N))
    w1probs <- 1./(1 + exp(-data %*% w1))
    w1probs <- cbind(w1probs,rep(1,N))
    w2probs <- 1./(1 + exp(-w1probs %*% w2))
    w2probs <- cbind(w2probs,rep(1,N))
    w3probs <- 1./(1 + exp(-w2probs %*% w3))
    w3probs <- cbind(w3probs,rep(1,N))
    w4probs <- w3probs %*% w4
    w4probs <- cbind(w4probs,rep(1,N))
    w5probs <- 1./(1 + exp(-w4probs %*% w5))
    w5probs <- cbind(w5probs,rep(1,N))
    w6probs <- 1./(1 + exp(-w5probs %*% w6))
    w6probs <- cbind(w6probs,rep(1,N))
    w7probs <- 1./(1 + exp(-w6probs %*% w7))
    w7probs <- cbind(w7probs,rep(1,N))
    dataout <- 1./(1 + exp(-w7probs %*% w8))
    err <- err + 1/N * sum((data[,1:(dim(data)[2]-1)] - dataout)^2)
  }
  test_err[epoch] <- err/testnumbatches
  
  data <- 0
  tt <- 0
  
  print("starting MINIMIZE.R")
  
  for (batch in 1:(numbatches/10)){
    print(c(epoch,batch))
    tt <- tt +1
    rm(data)
    data <- batchdata[,,(tt-1)*10+1]
    for (kk in 2:10){
      data <- rbind(data,batchdata[,,(tt-1)*10+kk])
    }
    
    # perform conjugate gradient with 3 linesearches
    
    max_iter <- 3
    
    VV <- c(w1, w2, w3, w4, w5, w6, w7, w8)
    Dim = c(l1, l2, l3, l4, l5, l6, l7, l8, l9)
    
    ret <- minimize(VV,cg_mnist,max_iter,Dim,data)
    
    X <- ret[[1]]
    
    w1 <- matrix(X[1: ((l1+1)*l2)], nrow = l1+1, ncol = l2)
    xxx = (l1+1) * l2
    w2 <- matrix(X[(xxx +1) : (xxx + (l2+1)*l3)], nrow = l2+1, ncol = l3)
    xxx = xxx + (l2+1) * l3
    w3 <- matrix(X[(xxx +1) : (xxx + (l3+1)*l4)], nrow = l3+1, ncol = l4)
    xxx = xxx + (l3+1) * l4
    w4 <- matrix(X[(xxx +1) : (xxx + (l4+1)*l5)], nrow = l4+1, ncol = l5)
    xxx = xxx + (l4+1) * l5
    w5 <- matrix(X[(xxx +1) : (xxx + (l5+1)*l6)], nrow = l5+1, ncol = l6)
    xxx = xxx + (l5+1) * l6
    w6 <- matrix(X[(xxx +1) : (xxx + (l6+1)*l7)], nrow = l6+1, ncol = l7)
    xxx = xxx + (l6+1) * l7
    w7 <- matrix(X[(xxx +1) : (xxx + (l7+1)*l8)], nrow = l7+1, ncol = l8)
    xxx = xxx + (l7+1) * l8
    w8 <- matrix(X[(xxx +1) : (xxx + (l8+1)*l9)], nrow = l8+1, ncol = l9)
    
  }
  
  save(w1,w2,w3,w4,w5,w6,w7,w8,file = paste("mnist_weights",epoch,".RData",sep = "_"))
  save(test_err,train_err,file = paste("mnist_error", epoch, ".RData", sep = "_"))
}

