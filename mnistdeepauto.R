rm(list=ls())

maxepoch <- 10 # set this to 10 when formally run program
numhid <- 1000
numpen <- 500
numpen2 <- 250
numopen <- 30

setwd("H:/HintonAutoencoder/Rcode")

# preload functions
source("converterR.r")
source("makebatchesR.r")

print('loading mnist dataset')
load_mnist()

print('makebatches')

batchdata <- makebatches()
testbatchdata <- maketestbatches()

set.seed(Sys.time())


print('Frist layer of RBM')
restart <- 1
source("rbmR.r")
hidrecbiases <- hidbiases
save(vishid,hidrecbiases,visbiases,file = "mnistvh.RData")
save(batchposhidprobs,file = "hidprob1.RData")

print('Second layer of RBM')
batchdata <- batchposhidprobs
numhid <- numpen
restart <- 1
source("rbmR.r")
hidpen <- vishid
penrecbiases <- hidbiases
hidgenbiases <- visbiases
save(hidpen,penrecbiases,hidgenbiases,file = "mnisthp.RData")
save(batchposhidprobs,file = "hidprob2.RData")

print('Third layer of RBM')
batchdata <- batchposhidprobs
numhid <- numpen2
restart <- 1
source("rbmR.r")
hidpen2 <- vishid
penrecbiases2 <- hidbiases
hidgenbiases2 <- visbiases
save(hidpen2,penrecbiases2,hidgenbiases2,file = "mnisthp2.RData")
save(batchposhidprobs,file = "hidprob3.RData")

print('Fourth layer of RBM')
batchdata <- batchposhidprobs
numhid <- numopen
restart <- 1
source("rbmhidlinearR.r")
hidtop <- vishid
toprecbiases <- hidbiases
topgenbiases <- visbiases
save(hidtop,toprecbiases,topgenbiases, file = "mnistpo.RData")
save(batchposhidprobs,file = "hidprob4.RData")