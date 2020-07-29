

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file(paste(getwd(),'/mnist/train-images.idx3-ubyte',sep=""))
  testr <<- load_image_file(paste(getwd(),'/mnist/t10k-images.idx3-ubyte',sep=""))
  
  train$y <<- load_label_file(paste(getwd(),'/mnist/train-labels.idx1-ubyte',sep=""))
  testr$y <<- load_label_file(paste(getwd(),'/mnist/t10k-labels.idx1-ubyte',sep=""))  
}

load_mnist()

darchmnist  <- darch(train$x,
                train$x,
                rbm.numEpochs = 0,
                rbm.batchSize = 100,
                rbm.lastLayer = F,
                layers = c(784,1000,500,250,2,250,500,1000,784),
                darch.batchSize = 100,
                retainData = F )


save(darchmnist,file = "darchmnist60k.RData")
