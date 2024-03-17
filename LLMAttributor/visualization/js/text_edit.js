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
        addingElement.classList.remove("left-adding-word")

        if (event.target.classList.contains("word")) {
            let clickedElement = event.target;
            if (document.getElementById("add-button").classList.contains("activated-button")) add_text(event, clickedElement);
        }
    }
    if (d3.select(".user-input-wrapper").style("display")!="none" && document.getElementById("edit-button").classList.contains("activated-button")) {
        let addingElement = document.getElementsByClassName("text-wrapper")[0].addingElement;
        if (!document.getElementsByClassName("text-wrapper")[0].adding) return;
        if (!document.getElementsByClassName("text-wrapper")[0].editAdding) return;
        if (event.target.id == addingElement.id) return;
        if (event.target.classList.contains("edit-hovered-word") ||event.target.classList.contains("editing-deleted-word") ) return;

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
        d3.selectAll(".word").classed("edit-hovered-word", false)
        d3.selectAll(".word").classed("edit-left-adding-word", false)
    }
})

document.addEventListener("mouseup", function(event) {
    if (document.getElementsByClassName("text-wrapper")[0].deleting) {
        document.getElementsByClassName("text-wrapper")[0].deleting = false;
        document.getElementsByClassName("text-wrapper")[0].deleteFromElement = null;
    }
    else if (document.getElementsByClassName("text-wrapper")[0].editDeleting) {
        document.getElementsByClassName("text-wrapper")[0].editDeleting = false;
        document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement = null;
    }
})

document.addEventListener("keydown", function(event) {
    if (d3.select(".user-input-wrapper").style("display")!="none" && event.key=="Enter") submitAddedText();
})

d3.selectAll(".button")
    .on("click", function(event) {
        if (this.id == "reset-button") {
            d3.select(".text-wrapper").html(document.getElementsByClassName("text-wrapper")[0].origianl_text_html);
            setWordEvents();
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
            if (this.id=="add-button") d3.select("body").style("cursor", 'url("./LLMAttributor/visualization/cursors/add-cursor.png") 16 16, zoom-in')
            if (this.id=="delete-button") d3.select("body").style("cursor", 'url("./LLMAttributor/visualization/cursors/delete-cursor.png") 16 16, zoom-in')
            if (this.id=="edit-button") d3.select("body").style("cursor", 'url("./LLMAttributor/visualization/cursors/edit-cursor.png") 16 16, zoom-in')
        }
        else d3.select("body").style("cursor", "default")
    })

function setWordEvents(){
    d3.selectAll(".word")
        .on("mouseover", function(event) {
            if (document.getElementById("add-button").classList.contains("activated-button")) {
                d3.select(this).classed("left-adding-word", true)
            }
            else if (document.getElementById("delete-button").classList.contains("activated-button")) {
                if (document.getElementsByClassName("text-wrapper")[0].deleting) {
                    let startElement = document.getElementsByClassName("text-wrapper")[0].deleteFromElement;
                    let tokenList = document.getElementsByClassName("word")
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

                        tokenList[i].classList.remove("delete-hovered-word")
                        if (deleteFlag && !deleteCancelling) tokenList[i].classList.add("deleted-word")
                        else if (deleteFlag && deleteCancelling) tokenList[i].classList.remove("deleted-word")
                        else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) tokenList[i].classList.add("deleted-word")
                        else tokenList[i].classList.remove("deleted-word")

                        if (tokenList[i]==startElement && !startLeftEndRight) deleteFlag = false;
                        if (tokenList[i]==this && startLeftEndRight) deleteFlag = false;

                    }
                }
                this.classList.add("delete-hovered-word")
            }
            else if (document.getElementById("edit-button").classList.contains("activated-button")) {
                if (this.classList.contains("deleted-word") && !this.classList.contains("edit-hovered-word")) return;
                if (document.getElementsByClassName("text-wrapper")[0].editAdding) return;
                if (document.getElementsByClassName("text-wrapper")[0].editDeleting) {
                    let startElement = document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement;
                    if (startElement==this) return;

                    let tokenList = document.getElementsByClassName("word")
                    
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

                        if (deleteFlag) {
                            tokenList[i].classList.add("deleted-word")
                            tokenList[i].classList.add("edit-hovered-word")
                        }
                        else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) {
                            tokenList[i].classList.add("deleted-word")
                        }
                        else {
                            tokenList[i].classList.remove("deleted-word")
                            tokenList[i].classList.remove("edit-hovered-word")
                        }

                        if (tokenList[i]==startElement && !startLeftEndRight) deleteFlag = false;
                        if (tokenList[i]==this && startLeftEndRight) deleteFlag = false;
                    }
                }
                else {
                    d3.select(this).classed("edit-left-adding-word", true)
                }
                this.classList.add("edit-hovered-word")
            }
        }).on("mouseout", function(event) {
            if (document.getElementsByClassName("text-wrapper")[0].adding && document.getElementsByClassName("text-wrapper")[0].addingElement==this) return;
            if (document.getElementById("add-button").classList.contains("activated-button")) {
                d3.select(this).classed("left-adding-word", false)
            }
            else if (document.getElementById("delete-button").classList.contains("activated-button")) {
                this.classList.remove("delete-hovered-word")
            }
            else if (document.getElementById("edit-button").classList.contains("activated-button")) {
                if (this.classList.contains("deleted-word")) return;
                if (document.getElementsByClassName("text-wrapper")[0].editDeleting) return;
                if (document.getElementsByClassName("text-wrapper")[0].editAdding) return;
                this.classList.remove("edit-hovered-word")
                this.classList.remove("edit-left-adding-word")
            }
        }).on("mousedown", function(event) {
            if (document.getElementsByClassName("text-wrapper")[0].adding) return;
            if (document.getElementById("delete-button").classList.contains("activated-button")) delete_text_start(event,this);
            else if (document.getElementById("edit-button").classList.contains("activated-button")) {
                if (this.classList.contains("deleted-word")) return;
                edit_text_start(event, this);
            }
        }).on("mouseup", function(event) {
            if (document.getElementsByClassName("text-wrapper")[0].adding) return;
            if (document.getElementById("delete-button").classList.contains("activated-button")) delete_text_done(event,this);
            else if (document.getElementById("edit-button").classList.contains("activated-button")) {
                if (this.classList.contains("deleted-word") && !this.classList.contains("edit-hovered-word")) return;
                edit_text_delete_done(event, this);
            }
        }).on("click", function(event) {
            if (document.getElementsByClassName("text-wrapper")[0].adding) return;
            if (document.getElementById("add-button").classList.contains("activated-button")) add_text(event, this);
        })
}

setWordEvents();

function add_text(event, clickedElement) {
    document.getElementsByClassName("text-wrapper")[0].adding = true;
    document.getElementsByClassName("text-wrapper")[0].addingElement = clickedElement;
    let tokenRect = clickedElement.getBoundingClientRect();
    let tokenRectLeft = tokenRect.left;
    let tokenRectBottom = tokenRect.bottom;
    d3.select(".user-input-wrapper")
        .style("display", "block")
        .style("left", `${tokenRectLeft-8}px`)
        .style("top", `${tokenRectBottom+3}px`)
}

function delete_text_start(event, clickedElement) {
    document.getElementsByClassName("text-wrapper")[0].deleting = true;
    document.getElementsByClassName("text-wrapper")[0].deleteCancelling = false;
    document.getElementsByClassName("text-wrapper")[0].deleteFromElement = clickedElement;
    if (clickedElement.classList.contains("deleted-word")) document.getElementsByClassName("text-wrapper")[0].deleteCancelling = true;

    // Save current delete status so that we can return to the original one when random mouseup happens 
    let currentDeleteStatus = []
    let tokenList = document.getElementsByClassName("word")
    for (let i=0; i<tokenList.length; i++) {
        currentDeleteStatus.push(tokenList[i].classList.contains("deleted-word"))
    }
    document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = currentDeleteStatus;

    if (document.getElementsByClassName("text-wrapper")[0].deleteCancelling) clickedElement.classList.remove("deleted-word")
    else clickedElement.classList.add("deleted-word")
}

function delete_text_done(event, endElement){
    // Check from the startElement to the endElement, and delete all the tokens in between
    let startElement = document.getElementsByClassName("text-wrapper")[0].deleteFromElement;
    let tokenList = document.getElementsByClassName("word")
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
        
        tokenList[i].classList.remove("delete-hovered-word")
        if (deleteFlag && !deleteCancelling) tokenList[i].classList.add("deleted-word")
        else if (deleteFlag && deleteCancelling) tokenList[i].classList.remove("deleted-word")
        else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) tokenList[i].classList.add("deleted-word")
        else tokenList[i].classList.remove("deleted-word")

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
    let tokenList = document.getElementsByClassName("word")
    for (let i=0; i<tokenList.length; i++) {
        currentDeleteStatus.push(tokenList[i].classList.contains("deleted-word"))
    }
    document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = currentDeleteStatus;

    clickedElement.classList.add("deleted-word")
}

function edit_text_delete_done(event, endElement) {
    // Check from the startElement to the endElement, and delete all the tokens in between
    let startElement = document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement;
    let tokenList = document.getElementsByClassName("word")
    
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
        
        tokenList[i].classList.remove("delete-hovered-word")
        if (deleteFlag) tokenList[i].classList.add("deleted-word")
        else if (document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus[i]) tokenList[i].classList.add("deleted-word")
        else tokenList[i].classList.remove("deleted-word")

        if (tokenList[i]==startElement && !startLeftEndRight) deleteFlag = false;
        if (tokenList[i]==endElement && startLeftEndRight) deleteFlag = false;
    }
    
    document.getElementsByClassName("text-wrapper")[0].editDeleting = false;
    document.getElementsByClassName("text-wrapper")[0].editDeleteFromElement = null;
    document.getElementsByClassName("text-wrapper")[0].prevDeleteStatus = null;

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
}

function submitAddedText() {
    let addedText = d3.select(".user-input").node().value;
    document.getElementsByClassName("user-input")[0].value = "";
    let addedTextNum = document.getElementsByClassName("text-wrapper")[0].addedTextNum
    let addingElement = document.getElementsByClassName("text-wrapper")[0].addingElement;
    
    let newNode = document.createElement("div");
    newNode.setAttribute("class", `word added-text added-text-${addedTextNum}`);

    let newContent = document.createTextNode(addedText);
    newNode.appendChild(newContent);
    document.getElementsByClassName("words-container")[0].insertBefore(newNode, addingElement)

    // After submitting the input, remove the margin-left of the clicked element, change "adding" to false
    document.getElementsByClassName("text-wrapper")[0].adding = false;
    document.getElementsByClassName("text-wrapper")[0].editAdding = false;
    document.getElementsByClassName("text-wrapper")[0].addingElement = null;
    document.getElementsByClassName("text-wrapper")[0].addedTextNum += 1;
    d3.select(".user-input-wrapper").style("display", "none")
    addingElement.classList.remove("left-adding-word")

    d3.selectAll(".word").classed("edit-left-adding-word", false)
    d3.selectAll(".word").classed("edit-hovered-word", false)
}

d3.select(".user-input-submit-button").on("click", submitAddedText)

d3.select(".copy-button").on("click", function(event) {
    let html = "<div>"
    let tokenList = document.getElementsByClassName("word")
    for (let i=0; i<tokenList.length; i++) {
        if (tokenList[i].classList.contains("dummy-word")) continue;
        if (tokenList[i].classList.contains("deleted-word")) {
            html += `<span class="llm-attributor-deleted-text">${tokenList[i].innerText}</span>`
        }
        else if (tokenList[i].classList.contains("added-text")) {
            html += `<span class="llm-attributor-added-text">${tokenList[i].innerText}</span>`
        }
        else html += (" " + tokenList[i].innerText)
    }
    html += "</div>"
    d3.select(".result-container").html(html)

    navigator.clipboard.writeText(`'${html}'`);
})