
epsilonw      <- 0.001 
epsilonvb     <- 0.001
epsilonhb     <- 0.001  
weightcost  <- 0.0002  
initialmomentum  <- 0.5
finalmomentum    <- 0.9

numcases <- dim(batchdata)[1]
numdims <- dim(batchdata)[2]
numbatches <- dim(batchdata)[3]

if (restart == 1){
  restart <- 0
  epoch <- 1
  
  vishid <- 0.1* matrix(runif(numdims*numhid),nrow = numdims, ncol = numhid)
  hidbiases <- matrix(rep(0,numhid),nrow = 1, ncol = numhid)
  visbiases <- matrix(rep(0,numdims),nrow = 1, ncol = numdims)
  
  poshidprobs <- matrix(rep(0,numdims*numhid),nrow = numdims, ncol = numhid)
  neghidprobs <- matrix(rep(0,numdims*numhid),nrow = numdims, ncol = numhid)
  posprobs <- matrix(rep(0,numdims*numhid),nrow = numdims, ncol = numhid)
  negprobs <- matrix(rep(0,numdims*numhid),nrow = numdims, ncol = numhid)
  vishidinc <- matrix(rep(0,numdims*numhid),nrow = numdims, ncol = numhid)
  hidbiasinc <- matrix(rep(0,numhid),nrow = 1, ncol = numhid)
  visbiasinc <- matrix(rep(0,numdims),nrow = 1, ncol = numdims)
  batchposhidprobs <- array(rep(0,numcases*numhid*numbatches),c(numcases,numhid,numbatches))
}

for (epoch in epoch:maxepoch){
  
  errsum <- 0
  for (batch in 1:numbatches){
    
    print(c(epoch,batch))
    
    # Start positive phase
    
    data <- batchdata[,,batch]
    hidbiasesMatrix <- matrix(rep(hidbiases,numcases),nrow = numcases, ncol = numhid)
    poshidprobs <- data %*% vishid + hidbiasesMatrix
    batchposhidprobs[,,batch] <- poshidprobs
    posprods <- t(data) %*% poshidprobs
    poshidact <- colSums(poshidprobs)
    posvisact <- colSums(data)
    
    # end of positive phase
    poshidstates <- (poshidprobs + matrix(runif(numhid *numcases),nrow = numcases, ncol = numhid))
    
    # start negative phase
    visbiasesMatrix <- matrix(rep(visbiases,numcases),nrow = numcases, ncol = numdims)
    negdata <- 1./(1 + exp(-poshidstates %*% t(vishid) - visbiasesMatrix))
    neghidprobs <- negdata %*% vishid + hidbiasesMatrix
    negprods <- t(negdata) %*% neghidprobs
    neghidact <- colSums(neghidprobs)
    negvisact <- colSums(negdata)
    
    # end of negative phase
    err <- sum((data-negdata)^2)
    errsum <- err + errsum
    
    if (epoch > 5){
      momentum <- finalmomentum
    }
    else{
      momentum <- initialmomentum
    }
    
    # update weights and biases
    vishidinc <- momentum * vishidinc + epsilonw*((posprods-negprods)/numcases - weightcost * vishid)
    visbiasinc <- momentum * visbiasinc + (epsilonvb/numcases) * (posvisact - negvisact)
    hidbiasinc <- momentum * hidbiasinc + (epsilonhb/numcases) * (poshidact - neghidact)
    
    vishid <- vishid + vishidinc;
    visbiases <- visbiases + visbiasinc
    hidbiases <- hidbiases + hidbiasinc
    
    # end of updates
  }
}