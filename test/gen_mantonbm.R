library(harmony)

metadata <- read.table("./data/MantonBM/metadata.csv", header = TRUE, sep = ',')
X <- read.table("./data/MantonBM/pca.txt")

start <- Sys.time()
Z <- HarmonyMatrix(X, metadata, "Channel", do_pca = FALSE)
end <- Sys.time()

write.table(Z, file = "./result/MantonBM_harmony_z.txt", quote = FALSE, row.names = FALSE, col.names = FALSE)

print(end - start)