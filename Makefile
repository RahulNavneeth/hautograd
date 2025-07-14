GHC = ghc
GHC_FLAGS = -Wall -O -isrc -ilib -odir $(BIN) -hidir $(BIN)
GHC_PKGS = vector
GHC_PKGS := $(addprefix -package , $(GHC_PKGS))

BIN = bin
SRC = $(shell find src -name '*.hs')
LIB = $(shell find lib -name '*.hs')
ALL_HS = $(SRC) $(LIB)
EXEC = $(BIN)/main

.PHONY: build clean run

build: $(EXEC)

$(EXEC): $(ALL_HS)
	mkdir -p $(BIN) 
	$(GHC) $(GHC_FLAGS) $(GHC_PKGS) -o $@ $^

clean:
	rm -rf $(BIN)

run: $(EXEC)
	@./$(EXEC)
