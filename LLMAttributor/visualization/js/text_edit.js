document.getElementsByClassName("text-wrapper")[0].origianl_text_html = d3.select(".text-wrapper").html();
document.getElementsByClassName("text-wrapper")[0].addedTextNum = 0;

document.addEventListener("mousedown", function(event) {
    if (d3.select(".user-input-wrapper").style("display")!="none" && document.getElementById("add-button").classList.contains("activated-button")) {
        let addingElement = document.getElementsByClassName("text-wrapper")[0].addingElement;
        if (event.target.id == addingElement.id) return;

        let userInputRect = document.getElementsByClassName("user-input-wrapper")[0].getBoundingClientRect()
        let buttonsRect = document.getElementsByClassName("buttons-wrapper")[0].getBoundingClientRect()
        let userInputLeft = userInputRect.x, userInputRight = userInputRect.x + userInputRect.width, userInputTop = userInputRect.y, userInputBottom = userInputRect.y + userInputRect.height
        let buttonsLeft = buttonsRect.x, buttonsRight = buttonsRect.x + buttonsRect.width, buttonsTop = buttonsRect.y, buttonsBottom = buttonsRect.y + buttonsRect.height

        if (event.clientX > userInputLeft && event.clientX < userInputRight && event.clientY > userInputTop && event.clientY < userInputBottom) return;
        if (event.clientX > buttonsLeft && event.clientX < buttonsRight && event.clientY > buttonsTop && event.clientY < buttonsBottom) return;
        
        document.getElementsByClassName("user-input")[0].value = "";
        d3.select(".user-input-wrapper").style("display", "none")
        document.getElementsByClassName("text-wrapper")[0].adding = false;
        document.getElementsByClassName("text-wrapper")[0].addingElement = null;
        let originalMargin = parseFloat(d3.select(addingElement).style("margin-left"))
        let originalPadding= parseFloat(d3.select(addingElement).style("padding-left"))
        d3.select(addingElement).style("background-color", "").style("margin-left", `${originalMargin-5}px`).style("padding-left", `${originalPadding-8}px`)

        if (event.target.classList.contains("token")) {
            let clickedElement = event.target;
            if (document.getElementById("add-button").classList.contains("activated-button")) add_text(event, clickedElement);
        }
    }
    if (d3.select(".user-input-wrapper").style("display")!="none" && document.getElementById("edit-button").classList.contains("activated-button")) {
        let addingElement = document.getElementsByClassName("text-wrapper")[0].addingElement;
        if (!document.getElementsByClassName("text-wrapper")[0].adding) return;
        if (!document.getElementsByClassName("text-wrapper")[0].editAdding) return;
        if (event.target.id == addingElement.id) return;
        if (event.target.classList.contains("edit-hovered-token") ||event.target.classList.contains("editing-deleted-token") ) return;

        let userInputRect = document.getElementsByClassName("user-input-wrapper")[0].getBoundingClientRect()
        let buttonsRect = document.getElementsByClassName("buttons-wrapper")[0].getBoundingClientRect()
        let userInputLeft = userInputRect.x, userInputRight = userInputRect.x + userInputRect.width, userInputTop = userInputRect.y, userInputBottom = userInputRect.y + userInputRect.height
        let buttonsLeft = buttonsRect.x, buttonsRight = buttonsRect.x + buttonsRect.width, buttonsTop = buttonsRect.y, buttonsBottom = buttonsRect.y + buttonsRect.height

        if (event.clientX > userInputLeft && event.clientX < userInputRight && event.clientY > userInputTop && event.clientY < userInputBottom) return;
        if (event.clientX > buttonsLeft && event.clientX < buttonsRight && event.clientY > buttonsTop && event.clientY < buttonsBottom) return;
        
        document.getElementsByClassName("user-input")[0].value = "";
        d3.select(".user-input-wrapper").style("display", "none")
        document.getElementsByClassName("text-wrapper")[0].adding = false;
        document.getElementsByClassName("text-wrapper")[0].editAdding = false;
        document.getElementsByClassName("text-wrapper")[0].addingElement = null;
        let originalMargin = parseFloat(d3.select(addingElement).style("margin-left"))
        let originalPadding= parseFloat(d3.select(addingElement).style("padding-left"))
        d3.select(addingElement).style("background-color", "").style("margin-left", `${originalMargin-5}px`).style("padding-left", `${originalPadding-8}px`)
        d3.selectAll(".token").classed("edit-hovered-token", false)
    }
})

document.addEventListener("mouseup", function(event) {
    if (document.getElementsByClassName("text-wrapper")[0].deleting) {
        if (event.target.classList.contains("token")) return;
        document.getElementsByClassName("text-wrapper")[0].deleting = false;
        document.getElementsByClassName("text-wrapper")[0].deleteFromElement = null;
        
        // Revert to the original delete status
        let tokenList = document.getElementsByClassName("token")
        for (let i=0; i<tokenList.length; i++) {
            if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) tokenList[i].classList.add("deleted-token")
            else tokenList[i].classList.remove("deleted-token")
        }
        document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = null;

    }
    else if (document.getElementsByClassName("text-wrapper")[0].editDeleting) {
        console.log(event.target)
        console.log("General mouseup event", event.target.classList.contains("token"))
        if (event.target.classList.contains("token")) return;
        let editDeleteFromElement = document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement;
        document.getElementsByClassName("text-wrapper")[0].editDeleting = false;
        document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement = null;

        let tokenList = document.getElementsByClassName("token")
        for (let i=0; i<tokenList.length; i++) {
            if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) {
                tokenList[i].classList.add("deleted-token")
            }
            else {
                tokenList[i].classList.remove("deleted-token")
                tokenList[i].classList.remove("edit-hovered-token")
            }
        }
        document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = null;

        let originalMargin = parseFloat(d3.select(editDeleteFromElement).style("margin-left"))
        let originalPadding= parseFloat(d3.select(editDeleteFromElement).style("padding-left"))
        d3.select(editDeleteFromElement).style("margin-left", `${originalMargin-5}px`).style("padding-left", `${originalPadding-8}px`)
    }
})

document.addEventListener("keydown", function(event) {
    if (d3.select(".user-input-wrapper").style("display")!="none" && event.key=="Enter") submitAddedText();
})

d3.selectAll(".button")
    .on("click", function(event) {
        if (this.id == "reset-button") {
            console.log("HERE")
            d3.select(".text-wrapper").html(document.getElementsByClassName("text-wrapper")[0].origianl_text_html);
            return;
        }
        if (document.getElementById(this.id).activated) document.getElementById(this.id).activated = false; 
        else document.getElementById(this.id).activated = true;

        let buttonList = document.getElementsByClassName("button")
        for (let i=0; i<buttonList.length; i++) {
            if (buttonList[i].id != this.id) buttonList[i].classList.remove("activated-button");
            else if (this.classList.contains("activated-button")) this.classList.remove("activated-button");
            else this.classList.add("activated-button");
        }

        // should change cursor
        if (this.classList.contains("activated-button")) {
            if (this.id=="add-button") d3.select("body").style("cursor", 'url("./LLMAttributor/visualization/icons/add-cursor.png"), zoom-in')
            if (this.id=="delete-button") d3.select("body").style("cursor", 'url("./LLMAttributor/visualization/icons/delete-cursor.png"), zoom-in')
            if (this.id=="edit-button") d3.select("body").style("cursor", 'url("./LLMAttributor/visualization/icons/edit-cursor.png"), zoom-in')
        }
        else d3.select("body").style("cursor", "default")
    })

d3.selectAll(".token")
    .on("mouseover", function(event) {
        // if (document.getElementsByClassName("text-wrapper")[0].adding) return;
        if (document.getElementById("add-button").classList.contains("activated-button")) {
            let newMargin= parseFloat(d3.select(this).style("margin-left"))+5
            let newPadding= parseFloat(d3.select(this).style("padding-left"))+8
            d3.select(this).style("background-color", "#ff000040").style("margin-left", `${newMargin}px`).style("padding-left", `${newPadding}px`)
        }
        else if (document.getElementById("delete-button").classList.contains("activated-button")) {
            if (document.getElementsByClassName("text-wrapper")[0].deleting) {
                let startElement = document.getElementsByClassName("text-wrapper")[0].deleteFromElement;
                let tokenList = document.getElementsByClassName("token")
                let deleteCancelling = document.getElementsByClassName("text-wrapper")[0].deleteCancelling;
                
                deleteFlag = false;
                let startLeftEndRight = false;
                for (let i=0; i<tokenList.length; i++) {
                    if (tokenList[i]==startElement && !deleteFlag) {
                        startLeftEndRight = true;
                        deleteFlag = true;
                    }
                    else if (tokenList[i]==this && !deleteFlag) {
                        startLeftEndRight = false;
                        deleteFlag = true;
                    }

                    tokenList[i].classList.remove("delete-hovered-token")
                    if (deleteFlag && !deleteCancelling) tokenList[i].classList.add("deleted-token")
                    else if (deleteFlag && deleteCancelling) tokenList[i].classList.remove("deleted-token")
                    else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) tokenList[i].classList.add("deleted-token")
                    else tokenList[i].classList.remove("deleted-token")

                    if (tokenList[i]==startElement && !startLeftEndRight) deleteFlag = false;
                    if (tokenList[i]==this && startLeftEndRight) deleteFlag = false;

                }
            }
            this.classList.add("delete-hovered-token")
        }
        else if (document.getElementById("edit-button").classList.contains("activated-button")) {
            if (this.classList.contains("deleted-token") && !this.classList.contains("edit-hovered-token")) return;
            if (document.getElementsByClassName("text-wrapper")[0].editAdding) return;
            if (document.getElementsByClassName("text-wrapper")[0].editDeleting) {
                let startElement = document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement;
                if (startElement==this) return;

                let tokenList = document.getElementsByClassName("token")
                
                deleteFlag = false;
                let startLeftEndRight = false;
                for (let i=0; i<tokenList.length; i++) {
                    if (tokenList[i]==startElement && !deleteFlag) {
                        startLeftEndRight = true;
                        deleteFlag = true;
                    }
                    else if (tokenList[i]==this && !deleteFlag) {
                        startLeftEndRight = false;
                        deleteFlag = true;
                    }

                    // tokenList[i].classList.remove("edit-hovered-token")
                    if (deleteFlag) {
                        tokenList[i].classList.add("deleted-token")
                        tokenList[i].classList.add("edit-hovered-token")
                    }
                    else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) {
                        tokenList[i].classList.add("deleted-token")
                    }
                    else {
                        tokenList[i].classList.remove("deleted-token")
                        tokenList[i].classList.remove("edit-hovered-token")
                    }

                    if (tokenList[i]==startElement && !startLeftEndRight) deleteFlag = false;
                    if (tokenList[i]==this && startLeftEndRight) deleteFlag = false;
                }
            }
            else {
                let newMargin= parseFloat(d3.select(this).style("margin-left"))+5
                let newPadding= parseFloat(d3.select(this).style("padding-left"))+8
                d3.select(this).style("margin-left", `${newMargin}px`).style("padding-left", `${newPadding}px`)
            }
            this.classList.add("edit-hovered-token")
        }
    }).on("mouseout", function(event) {
        if (document.getElementsByClassName("text-wrapper")[0].adding && document.getElementsByClassName("text-wrapper")[0].addingElement==this) return;
        if (document.getElementById("add-button").classList.contains("activated-button")) {
            let originalMargin = parseFloat(d3.select(this).style("margin-left"))
            let originalPadding= parseFloat(d3.select(this).style("padding-left"))
            d3.select(this).style("background-color", "").style("margin-left", `${originalMargin-5}px`).style("padding-left", `${originalPadding-8}px`)
        }
        else if (document.getElementById("delete-button").classList.contains("activated-button")) {
            this.classList.remove("delete-hovered-token")
        }
        else if (document.getElementById("edit-button").classList.contains("activated-button")) {
            if (this.classList.contains("deleted-token")) return;
            if (document.getElementsByClassName("text-wrapper")[0].editDeleting) return;
            if (document.getElementsByClassName("text-wrapper")[0].editAdding) return;
            this.classList.remove("edit-hovered-token")
            // if (document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement==this) return;

            // this.classList.remove("delete-hovered-token")
            let originalMargin = parseFloat(d3.select(this).style("margin-left"))
            let originalPadding= parseFloat(d3.select(this).style("padding-left"))
            d3.select(this).style("background-color", "").style("margin-left", `${originalMargin-5}px`).style("padding-left", `${originalPadding-8}px`)
        }
    }).on("mousedown", function(event) {
        if (document.getElementsByClassName("text-wrapper")[0].adding) return;
        if (document.getElementById("delete-button").classList.contains("activated-button")) delete_text_start(event,this);
        else if (document.getElementById("edit-button").classList.contains("activated-button")) {
            if (this.classList.contains("deleted-token")) return;
            edit_text_start(event, this);
        }
    }).on("mouseup", function(event) {
        if (document.getElementsByClassName("text-wrapper")[0].adding) return;
        if (document.getElementById("delete-button").classList.contains("activated-button")) delete_text_done(event,this);
        else if (document.getElementById("edit-button").classList.contains("activated-button")) {
            if (this.classList.contains("deleted-token") && !this.classList.contains("edit-hovered-token")) return;
            edit_text_delete_done(event, this);
        }
    }).on("click", function(event) {
        if (document.getElementsByClassName("text-wrapper")[0].adding) return;
        if (document.getElementById("add-button").classList.contains("activated-button")) add_text(event, this);
    })

function add_text(event, clickedElement) {
    console.log("add_text", clickedElement)
    document.getElementsByClassName("text-wrapper")[0].adding = true;
    document.getElementsByClassName("text-wrapper")[0].addingElement = clickedElement;
    let tokenRect = clickedElement.getBoundingClientRect();
    let tokenRectLeft = tokenRect.left;
    let tokenRectBottom = tokenRect.bottom;
    console.log(tokenRect)
    d3.select(".user-input-wrapper")
        .style("display", "block")
        .style("left", `${tokenRectLeft-8}px`)
        .style("top", `${tokenRectBottom+3}px`)
}

function delete_text_start(event, clickedElement) {
    document.getElementsByClassName("text-wrapper")[0].deleting = true;
    document.getElementsByClassName("text-wrapper")[0].deleteCancelling = false;
    document.getElementsByClassName("text-wrapper")[0].deleteFromElement = clickedElement;
    if (clickedElement.classList.contains("deleted-token")) document.getElementsByClassName("text-wrapper")[0].deleteCancelling = true;

    // Save current delete status so that we can return to the original one when random mouseup happens 
    let currentDeleteStatus = []
    let tokenList = document.getElementsByClassName("token")
    for (let i=0; i<tokenList.length; i++) {
        currentDeleteStatus.push(tokenList[i].classList.contains("deleted-token"))
    }
    document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = currentDeleteStatus;

    if (document.getElementsByClassName("text-wrapper")[0].deleteCancelling) clickedElement.classList.remove("deleted-token")
    else clickedElement.classList.add("deleted-token")
}

function delete_text_done(event, endElement){
    // Check from the startElement to the endElement, and delete all the tokens in between
    let startElement = document.getElementsByClassName("text-wrapper")[0].deleteFromElement;
    let tokenList = document.getElementsByClassName("token")
    let deleteCancelling = document.getElementsByClassName("text-wrapper")[0].deleteCancelling;
    
    deleteFlag = false;
    let startLeftEndRight = false;
    for (let i=0; i<tokenList.length; i++) {
        if (tokenList[i]==startElement && !deleteFlag) {
            startLeftEndRight = true;
            deleteFlag = true;
        }
        else if (tokenList[i]==endElement && !deleteFlag) {
            startLeftEndRight = false;
            deleteFlag = true;
        }
        
        tokenList[i].classList.remove("delete-hovered-token")
        if (deleteFlag && !deleteCancelling) tokenList[i].classList.add("deleted-token")
        else if (deleteFlag && deleteCancelling) tokenList[i].classList.remove("deleted-token")
        else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) tokenList[i].classList.add("deleted-token")
        else tokenList[i].classList.remove("deleted-token")

        if (tokenList[i]==startElement && !startLeftEndRight) deleteFlag = false;
        if (tokenList[i]==endElement && startLeftEndRight) deleteFlag = false;
    }
    
    document.getElementsByClassName("text-wrapper")[0].deleting = false;
    document.getElementsByClassName("text-wrapper")[0].deleteFromElement = null;
    document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = null;
    document.getElementsByClassName("text-wrapper")[0].deleteCancelling = null;
}

function edit_text_start(event, clickedElement) {
    document.getElementsByClassName("text-wrapper")[0].editDeleting = true;
    document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement = clickedElement;

    // Save current delete status so that we can return to the original one when random mouseup happens 
    let currentDeleteStatus = []
    let tokenList = document.getElementsByClassName("token")
    for (let i=0; i<tokenList.length; i++) {
        currentDeleteStatus.push(tokenList[i].classList.contains("deleted-token"))
    }
    document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = currentDeleteStatus;

    clickedElement.classList.add("deleted-token")
}

function edit_text_delete_done(event, endElement) {
    console.log("edit_text_delete_done", endElement)
    // Check from the startElement to the endElement, and delete all the tokens in between
    let startElement = document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement;
    let tokenList = document.getElementsByClassName("token")
    
    deleteFlag = false;
    let startLeftEndRight = false;
    for (let i=0; i<tokenList.length; i++) {
        if (tokenList[i]==startElement && !deleteFlag) {
            startLeftEndRight = true;
            deleteFlag = true;
        }
        else if (tokenList[i]==endElement && !deleteFlag) {
            startLeftEndRight = false;
            deleteFlag = true;
        }
        
        tokenList[i].classList.remove("delete-hovered-token")
        if (deleteFlag) tokenList[i].classList.add("deleted-token")
        else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) tokenList[i].classList.add("deleted-token")
        else tokenList[i].classList.remove("deleted-token")

        if (tokenList[i]==startElement && !startLeftEndRight) deleteFlag = false;
        if (tokenList[i]==endElement && startLeftEndRight) deleteFlag = false;
    }
    
    document.getElementsByClassName("text-wrapper")[0].editDeleting = false;
    document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement = null;
    document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = null;

    console.log("edit_text_delete_done -- change adding status")
    document.getElementsByClassName("text-wrapper")[0].editAdding = true;
    document.getElementsByClassName("text-wrapper")[0].adding = true;
    document.getElementsByClassName("text-wrapper")[0].addingElement = startElement;
    let tokenRect = startElement.getBoundingClientRect();
    let tokenRectLeft = tokenRect.left;
    let tokenRectBottom = tokenRect.bottom;
    d3.select(".user-input-wrapper")
        .style("display", "block")
        .style("left", `${tokenRectLeft-8}px`)
        .style("top", `${tokenRectBottom+3}px`)
    console.log("edit_text_delete_done - shown")
}

function submitAddedText() {
    let addedText = d3.select(".user-input").node().value;
    document.getElementsByClassName("user-input")[0].value = "";
    let addedTextNum = document.getElementsByClassName("text-wrapper")[0].addedTextNum
    let addingElement = document.getElementsByClassName("text-wrapper")[0].addingElement;
    console.log("submitAddedText", addingElement)
    
    let newNode = document.createElement("div");
    newNode.setAttribute("id", `added-text-${addedTextNum}`);

    if (addedText==" ") newNode.setAttribute("class", "added-text space-token")
    else if (addedText[0]==" ") {
        addedText = addedText.slice(1);
        newNode.setAttribute("class", "token added-text left-space-token")
    }
    else newNode.setAttribute("class", "token added-text");
    
    let newContent = document.createTextNode(addedText);
    newNode.appendChild(newContent);
    document.getElementsByClassName("tokens-container")[0].insertBefore(newNode, addingElement.parentNode)

    // After submitting the input, remove the margin-left of the clicked element, change "adding" to false
    document.getElementsByClassName("text-wrapper")[0].adding = false;
    document.getElementsByClassName("text-wrapper")[0].editAdding = false;
    document.getElementsByClassName("text-wrapper")[0].addingElement = null;
    document.getElementsByClassName("text-wrapper")[0].addedTextNum += 1;
    d3.select(".user-input-wrapper").style("display", "none")
    let originalMargin = parseFloat(d3.select(addingElement).style("margin-left"))
    let originalPadding= parseFloat(d3.select(addingElement).style("padding-left"))
    d3.select(addingElement).style("background-color", "").style("margin-left", `${originalMargin-5}px`).style("padding-left", `${originalPadding-8}px`)

    // TODO: for all elements, remove ".editing-deleted-token" class
    d3.selectAll(".token").classed("edit-hovered-token", false)
    // tokenList[i].classList.remove("edit-hovered-token")
}

d3.select(".user-input-submit-button").on("click", submitAddedText)

d3.select(".copy-button").on("click", function(event) {
    let html = "<div>"
    let tokenList = document.getElementsByClassName("token")
    for (let i=0; i<tokenList.length; i++) {
        if (tokenList[i].classList.contains("deleted-token")) continue;
        if (tokenList[i].classList.contains("dummy-token")) continue;
        if (tokenList[i].classList.contains("space-token")) html += " ";
        else if (tokenList[i].classList.contains("added-text") && tokenList[i].classList.contains("left-space-token")) {
            html += (" "+`<span class="llm-attributor-added-result-new">${tokenList[i].innerText}</span>`)
        }
        else if (tokenList[i].classList.contains("added-text")) html += `<span class="result-new">${tokenList[i].innerText}</span>`
        else if (tokenList[i].classList.contains("left-space-token")) {html += (" "+tokenList[i].innerText)}
        else html += tokenList[i].innerText
    }
    html += "</div>"
    d3.select(".result-container").html(html)

    navigator.clipboard.writeText(`'${html}'`);
})