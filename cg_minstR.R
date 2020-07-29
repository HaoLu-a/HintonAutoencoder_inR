cg_mnist <- function(VV, Dim, XX){
  # VV: starting point
  
  l1 <- Dim[1]
  l2 <- Dim[2]
  l3 <- Dim[3]
  l4 <- Dim[4]
  l5 <- Dim[5]
  l6 <- Dim[6]
  l7 <- Dim[7]
  l8 <- Dim[8]
  l9 <- Dim[9]
  N <- nrow(XX)
  
  w1 <- matrix(VV[1: ((l1+1)*l2)], nrow = l1+1, ncol = l2)
  xxx = (l1+1) * l2
  w2 <- matrix(VV[(xxx +1) : (xxx + (l2+1)*l3)], nrow = l2+1, ncol = l3)
  xxx = xxx + (l2+1) * l3
  w3 <- matrix(VV[(xxx +1) : (xxx + (l3+1)*l4)], nrow = l3+1, ncol = l4)
  xxx = xxx + (l3+1) * l4
  w4 <- matrix(VV[(xxx +1) : (xxx + (l4+1)*l5)], nrow = l4+1, ncol = l5)
  xxx = xxx + (l4+1) * l5
  w5 <- matrix(VV[(xxx +1) : (xxx + (l5+1)*l6)], nrow = l5+1, ncol = l6)
  xxx = xxx + (l5+1) * l6
  w6 <- matrix(VV[(xxx +1) : (xxx + (l6+1)*l7)], nrow = l6+1, ncol = l7)
  xxx = xxx + (l6+1) * l7
  w7 <- matrix(VV[(xxx +1) : (xxx + (l7+1)*l8)], nrow = l7+1, ncol = l8)
  xxx = xxx + (l7+1) * l8
  w8 <- matrix(VV[(xxx +1) : (xxx + (l8+1)*l9)], nrow = l8+1, ncol = l9)
  
  XX <- cbind(XX,rep(1,N))
  w1probs <- 1./(1 + exp(-XX %*% w1))
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
  XXout <- 1./(1 + exp(-w7probs %*% w8))
  
  f <- -1/N * sum(XX[,1:(ncol(XX)-1)] * log(XXout) + (1-XX[,1:(ncol(XX)-1)]) * log(1-XXout))
  IO <- 1/N * (XXout - XX[,1:(ncol(XX)-1)])
  Ix8 <- IO
  dw8 <- t(w7probs) %*% Ix8
  
  Ix7 <- (Ix8 %*% t(w8)) * w7probs * (1-w7probs)
  Ix7 <- Ix7[,1:ncol(Ix7)-1]
  dw7 <- t(w6probs) %*% Ix7
  
  Ix6 <- (Ix7 %*% t(w7)) * w6probs * (1-w6probs)
  Ix6 <- Ix6[,1:ncol(Ix6)-1]
  dw6 <- t(w5probs) %*% Ix6
  
  Ix5 <- (Ix6 %*% t(w6) * w5probs * (1-w5probs))
  Ix5 <- Ix5[,1:ncol(Ix5)-1]
  dw5 <- t(w4probs) %*% Ix5
  
  Ix4 <- (Ix5 %*% t(w5) * w4probs * (1-w4probs))
  Ix4 <- Ix4[,1:ncol(Ix4)-1]
  dw4 <- t(w3probs) %*% Ix4
  
  Ix3 <- (Ix4 %*% t(w4) * w3probs * (1-w3probs))
  Ix3 <- Ix3[,1:ncol(Ix3)-1]
  dw3 <- t(w2probs) %*% Ix3
  
  Ix2 <- (Ix3 %*% t(w3) * w2probs * (1-w2probs))
  Ix2 <- Ix2[,1:ncol(Ix2)-1]
  dw2 <- t(w1probs) %*% Ix2
  
  Ix1 <- (Ix2 %*% t(w2) * w1probs * (1-w1probs))
  Ix1 <- Ix1[,1:ncol(Ix1)-1]
  dw1 <- t(XX) %*% Ix1
  
  
  df <- c(dw1,dw2,dw3,dw4,dw5,dw6,dw7,dw8)
  
  return(c(f,df))
  
}