let tokensContainerId, tokensContainer;

document.addEventListener("selectAttrTokensPos", function(event) {
    console.log(event)
    tokensContainerId = event.token_container_id;
    tokensContainer = document.getElementById(tokensContainerId);
    tokensContainer.random_int_id = tokensContainerId.split("-")[2]
    tokensContainer.prompt_token_num = event.prompt_token_num;
    tokensContainer.total_token_num = event.total_token_num;
    tokensContainer.startToken = -1;
    tokensContainer.beingDragged = false;
    tokensContainer.selecting = false;
    tokensContainer.highlightColor = "#2171b5";
    tokensContainer.neutralColor = "#000000";
    tokensContainer.fadeColor = "#909090";
    for (tokenElement of document.getElementsByClassName("attended-token")) tokenElement.selected = false;

    d3.select(`#copy-selected-token-idx-button-${tokensContainer.random_int_id}`).on("click", function(event) {
        let highlightedTokenIndices = []; 
        for (let i=tokensContainer.prompt_token_num; i<tokensContainer.total_token_num; i++) {
            let tokenElement = document.getElementById(`token-${tokensContainer.random_int_id}-${i}`)
            if (i==0) continue; // as we cannot track the probability of the first token to be generated from nowhere
            if (tokenElement == null) continue;
            if (tokenElement.selected) highlightedTokenIndices.push(i - tokensContainer.prompt_token_num);
        }
    
        let highlightedTokenIndicesStr = "[" + highlightedTokenIndices.join(",") + "]";
        document.getElementById(`selected-token-indices-${tokensContainer.random_int_id}`).innerText = highlightedTokenIndicesStr;
        navigator.clipboard.writeText(highlightedTokenIndicesStr);
    })    
})

d3.selectAll(".attended-token").style("cursor", "pointer"); 

d3.selectAll(".attended-token").on("mousedown", function(event) {
    // let tokenElement = event.target 
    let tokenElement = this;
    let clickedTokenIdx = parseInt(tokenElement.id.split("-")[2]);

    if (clickedTokenIdx >= tokensContainer.prompt_token_num) {
        tokensContainer.startToken = clickedTokenIdx;
        tokensContainer.beingDragged = true;
        if (tokenElement.selected) {tokenElement.newSelected = false; tokensContainer.selecting=false;}
        else {tokenElement.newSelected = true; tokensContainer.selecting=true;}

        if (tokensContainer.selecting) tokenElement.classList.add("selected-token");
        else tokenElement.classList.remove("selected-token");
    }
})

// After dragging, mouseup can happen anywhere
document.addEventListener("mouseup", function(event) {
    if (tokensContainer.beingDragged) {{
        for (let i=tokensContainer.prompt_token_num; i<tokensContainer.total_token_num; i++) {
            let tokenElement = document.getElementById(`token-${tokensContainer.random_int_id}-${i}`);
            if (tokenElement==null) continue;
            tokenElement.selected = tokenElement.newSelected;
        }
    }}
    tokensContainer.startToken = -1;
    tokensContainer.beingDragged = false;
    tokensContainer.selecting = false;
})

d3.selectAll(".attended-token").on("mouseenter", function(event) {
    let enteredTokenElement = this
    let enteredTokenIdx = parseInt(enteredTokenElement.id.split("-")[2]);

    if (enteredTokenIdx >= tokensContainer.prompt_token_num) enteredTokenElement.style.backgroundColor = `${tokensContainer.highlightColor}60`;
    if (tokensContainer.beingDragged) {
        let start = Math.min(tokensContainer.startToken, enteredTokenIdx);
        let end = Math.max(tokensContainer.startToken, enteredTokenIdx);
        for (let i=tokensContainer.prompt_token_num; i<tokensContainer.total_token_num; i++) {
            let tokenElement = document.getElementById(`token-${tokensContainer.random_int_id}-${i}`)
            if (tokenElement==null) continue;
            if (i>=start && i<=end) {
                tokenElement.newSelected = tokensContainer.selecting;
                if (tokensContainer.selecting) tokenElement.classList.add("selected-token");
                else tokenElement.classList.remove("selected-token");
            }
            else {
                tokenElement.newSelected = tokenElement.selected;
                if (tokenElement.selected) tokenElement.classList.add("selected-token");
                else tokenElement.classList.remove("selected-token");
            }
        }
    }      
})

d3.selectAll(".token").on("mouseout", function(event) {
    let token = event.target;
    token.style.backgroundColor = `${tokensContainer.highlightColor}00`
})


