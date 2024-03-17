let gPositiveAttribution, gNegativeAttribution, uPositiveAttribution, uNegativeAttribution;
let gScoreHistogramBins, uScoreHistogramBins;
let gScoreHistogramCounts, uScoreHistogramCounts;
let gPosTfIdf, gNegTfIdf, uPosTfIdf, uNegTfIdf;
let positiveMaxNum, negativeMaxNum;
let num_g_pos_shown = 1
let num_u_pos_shown = 1
let num_g_neg_shown = 1
let num_u_neg_shown = 1
let expanded = false;
let expandedAttributedTextWrapperElementId = "";
let expandedAttributedTextElementId = "";
let histogramHighlightedTextWrapperElementIds = [];
let iframeId;

document.addEventListener("compare", function(event) {
    console.log(event)

    gPositiveAttribution = event.gPositiveAttribution
    gNegativeAttribution = event.gNegativeAttribution
    uPositiveAttribution = event.uPositiveAttribution
    uNegativeAttribution = event.uNegativeAttribution
    gScoreHistogramBins = event.gScoreHistogramBins
    uScoreHistogramBins = event.uScoreHistogramBins
    gScoreHistogramCounts = event.gScoreHistogramCounts
    uScoreHistogramCounts = event.uScoreHistogramCounts
    gPosTfIdf = event.gPosTfIdf
    gNegTfIdf = event.gNegTfIdf
    uPosTfIdf = event.uPosTfIdf
    uNegTfIdf = event.uNegTfIdf
    positiveMaxNum = event.positiveMaxNum
    negativeMaxNum = event.negativeMaxNum
    iframeId = event.iframeId

    firstTokenSpacing(d3.select(".user-provided-text"))
    firstTokenSpacing(d3.select(".generated-text"))

    addAttributionNumDropdown(true, true)
    addAttributionNumDropdown(true, false)
    addAttributionNumDropdown(false, true)
    addAttributionNumDropdown(false, false)

    displayAttribution(true, true)
    displayAttribution(true, false)
    displayAttribution(false, true)
    displayAttribution(false, false)

    displayTFIDF(true, true)
    displayTFIDF(true, false)
    displayTFIDF(false, true)
    displayTFIDF(false, false)

    drawHistogram()
    document.scrollingElement.style.scrollbarGutter = "stable"
    document.scrollingElement.style.scrollbarWidth = "thin"
})

function addAttributionNumDropdown(generated, positive) {
    let textType = generated ? "generated" : "user-provided"
    let positiveType = positive ? "positive" : "negative"
    let color = generated ? "blue":"red"
    
    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-wrapper`)
        .append("div")
        .attr("class", `attribution-num-dropdown-title-wrapper ${textType}-${positiveType}-attribution-num-dropdown-title-wrapper`)
        .style("border-bottom", `1px solid var(--${color}7)`)
        .on("mouseover", function(event) {  // Mouseover Event Listener
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-svg`).style("opacity", 0.7)
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-text`).style("opacity", 0.7)
        })
        .on("mouseout", function(event) {  // Mouseover Event Listener
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-svg`).style("opacity", 1)
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-text`).style("opacity", 1)
        })
        .on("click", function(event) {  // Click Event Listener
            d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`)
                .style("display", d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`).style("display")=="none"?"block":"none")  
            document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded = !document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded
        })
        .append("div")
            .attr("class", `attribution-num-dropdown-text ${textType}-${positiveType}-attribution-num-dropdown-text`)
            .text(positive ? (generated?num_g_pos_shown:num_u_pos_shown) : (generated?num_g_neg_shown:num_u_neg_shown))

    document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded = false

    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-title-wrapper`)
        .append("svg")
        .attr("class", `attribution-num-dropdown-svg ${textType}-${positiveType}-attribution-num-dropdown-svg`)
        .attr("width", 10)
        .attr("height", 10)
        .append("path")
            .attr("d", "M 0 0 L 10 0 L 5 10 Z")
            .attr("fill", `var(--${color}3)`)
    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-wrapper`)
        .append("div")
            .attr("class", `attribution-num-dropdown-options-wrapper ${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`)
            .style("display", "none")
            .selectAll("p")
            .data([...Array(positive?positiveMaxNum+1:negativeMaxNum+1).keys()])
            .enter()
            .append("p")
                .attr("class", `attribution-num-dropdown-option ${textType}-${positiveType}-attribution-num-dropdown-option`)
                .style("background-color", `var(--${color}0)`)
                .style("color", `var(--${color}4)`)
                .style("border", `1px solid var(--${color}1)`)
                .text(d => d)
                .on("mouseover", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", `var(--${color}1)`)
                })
                .on("mouseout", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", `var(--${color}0)`)
                })
                .on("click", function(event) {  // Click Event Listener
                    if (generated&&positive) num_g_pos_shown = +d3.select(this).text()
                    else if (generated&&!positive) num_g_neg_shown = +d3.select(this).text()
                    else if (!generated&&positive) num_u_pos_shown = +d3.select(this).text()
                    else num_u_neg_shown = +d3.select(this).text()
                    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-text`).text(d3.select(this).text())
                    displayAttribution(generated, positive)
                    d3.select(`.${textType}-${positiveType}-attribution-num-dropdown-options-wrapper`).style("display", "none")
                    document.getElementsByClassName(`${textType}-${positiveType}-attribution-num-dropdown-wrapper`)[0].expanded = false
                    displayTFIDF(generated, positive)
                })
}

document.addEventListener("mouseup", function(event) {
    let generatedPositiveAttributionNumDropdownWrapper = document.getElementsByClassName("generated-positive-attribution-num-dropdown-wrapper")[0]
    let generatedNegativeAttributionNumDropdownWrapper = document.getElementsByClassName("generated-negative-attribution-num-dropdown-wrapper")[0]
    let userPositiveAttributionNumDropdownWrapper = document.getElementsByClassName("user-provided-positive-attribution-num-dropdown-wrapper")[0]
    let userNegativeAttributionNumDropdownWrapper = document.getElementsByClassName("user-provided-negative-attribution-num-dropdown-wrapper")[0]

    let generatedPositiveDropdownBox = generatedPositiveAttributionNumDropdownWrapper.getBoundingClientRect()
    let generatedNegativeDropdownBox = generatedNegativeAttributionNumDropdownWrapper.getBoundingClientRect()
    let userPositiveDropdownBox = userPositiveAttributionNumDropdownWrapper.getBoundingClientRect()
    let userNegativeDropdownBox = userNegativeAttributionNumDropdownWrapper.getBoundingClientRect()

    if (event.clientX > generatedPositiveDropdownBox.left && event.clientX < generatedPositiveDropdownBox.right && event.clientY > generatedPositiveDropdownBox.top && event.clientY < generatedPositiveDropdownBox.bottom) return;
    if (event.clientX > generatedNegativeDropdownBox.left && event.clientX < generatedNegativeDropdownBox.right && event.clientY > generatedNegativeDropdownBox.top && event.clientY < generatedNegativeDropdownBox.bottom) return;
    if (event.clientX > userPositiveDropdownBox.left && event.clientX < userPositiveDropdownBox.right && event.clientY > userPositiveDropdownBox.top && event.clientY < userPositiveDropdownBox.bottom) return;
    if (event.clientX > userNegativeDropdownBox.left && event.clientX < userNegativeDropdownBox.right && event.clientY > userNegativeDropdownBox.top && event.clientY < userNegativeDropdownBox.bottom) return;

    if (generatedPositiveAttributionNumDropdownWrapper.expanded) {
        d3.select(".generated-positive-attribution-num-dropdown-options-wrapper").style("display", "none")
        generatedPositiveAttributionNumDropdownWrapper.expanded = false
    }
    if (generatedNegativeAttributionNumDropdownWrapper.expanded) {
        d3.select(".generated-negative-attribution-num-dropdown-options-wrapper").style("display", "none")
        generatedNegativeAttributionNumDropdownWrapper.expanded = false
    }
    if (userPositiveAttributionNumDropdownWrapper.expanded) {
        d3.select(".user-provided-positive-attribution-num-dropdown-options-wrapper").style("display", "none")
        userPositiveAttributionNumDropdownWrapper.expanded = false
    }
    if (userNegativeAttributionNumDropdownWrapper.expanded) {
        d3.select(".user-provided-negative-attribution-num-dropdown-options-wrapper").style("display", "none")
        userNegativeAttributionNumDropdownWrapper.expanded = false
    }

    // If attribution box is expanded and click is outside the box, collapse the box
    if (expanded) {
        let expandedBox = document.getElementById(expandedAttributedTextWrapperElementId).getBoundingClientRect()
        if (event.clientX < expandedBox.left || event.clientX > expandedBox.right || event.clientY < expandedBox.top || event.clientY > expandedBox.bottom) {
            collapseExpandedAttributedText(expandedAttributedTextWrapperElementId, expandedAttributedTextElementId)
            if (!event.target.classList.contains("attributed-text-wrapper")) {
                expanded = false;
                expandedAttributedTextWrapperElementId = "";
                expandedAttributedTextElementId = "";
            }
        }
    }
})

function displayAttribution(generated, positive) {
    let textType = generated ? "generated" : "user-provided"
    let positiveType = positive ? "positive" : "negative"
    let color = generated ? "blue":"red"
    let num_shown = positive ? (generated?num_g_pos_shown:num_u_pos_shown) : (generated?num_g_neg_shown:num_u_neg_shown)
    let attribution = positive ? (generated ? gPositiveAttribution : uPositiveAttribution) : (generated ? gNegativeAttribution : uNegativeAttribution)

    d3.select(`.${textType}-${positiveType}-attribution`).selectAll(".attributed-text-wrapper").remove()

    for (let idx=0 ; idx<num_shown ; idx++) {
        let i = positive?idx:num_shown-idx-1;
        if (attribution.length <= i) break;
        d3.select(`.${textType}-${positiveType}-attribution`)
            .append("div")
                .attr("class", `attributed-text-wrapper ${textType}-${positiveType}-attributed-text-wrapper`)
                .attr("id", `${textType}-${positiveType}-attributed-text-wrapper-${i}`)
                .style("background-color", `var(--${color}${positive?1:0})`)
                .style("border", `1px solid var(--${color}${positive?4:3})`)
                .style("color", `var(--${color}${positive?6:5})`)

        let attributedTextWrapperElement = document.getElementById(`${textType}-${positiveType}-attributed-text-wrapper-${i}`)
        attributedTextWrapperElement.attributionData = attribution[i]
        attributedTextWrapperElement.positive = positive 
        attributedTextWrapperElement.generated = generated

        d3.select(`#${textType}-${positiveType}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("class", `attributed-text-info-wrapper ${textType}-${positiveType}-attributed-text-info-wrapper`)
                .attr("id", `${textType}-${positiveType}-attributed-text-info-wrapper-${i}`)
        d3.select(`#${textType}-${positiveType}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("class", `attributed-text ${textType}-${positiveType}-attributed-text`)
                .attr("id", `${textType}-${positiveType}-attributed-text-${i}`)
        setCollapsedText(generated, positive, i, attribution[i]) 

        d3.select(`.${textType}-${positiveType}-attribution`)
            .selectAll(".attributed-text-wrapper")
            .on("click", function(event) {  // Click Event Listener
                let clickedWrapperId = event.target.id 
                let clickedWrapperElement = document.getElementById(clickedWrapperId)
                if (expanded && clickedWrapperElement.expanded) return;

                let clickedAttributedTextElementId = d3.select(`#${clickedWrapperId}`).select(`.attributed-text`).attr("id")
                let data = clickedWrapperElement.attributionData
                d3.select(`#${clickedWrapperId}`).select(".attributed-text").remove()

                d3.select(`#${clickedWrapperId}`)
                    .append("div")
                    .attr("class", "expanded-attributed-text-wrapper")
                    .style("opacity", "0")
                    .transition()
                    .duration(1000)
                        .style("opacity", "1")
                    
                d3.select(`#${clickedWrapperId}`)
                    .select(".expanded-attributed-text-wrapper")
                    .append("div")
                        .attr("class", "expanded-attributed-text-full-text-wrapper expanded-attributed-text-contents")
                        .html(data["text_html_code"])

                let tokensContainerId = d3.select(`#${clickedWrapperId}`).select(".tokens-container").attr("id")
                firstTokenSpacing(d3.select(`#${tokensContainerId}`))

                for (let key in data) {
                    if (key=="text_html_code"||key=="text"||key=="score_histogram_bin"||key=="tokens_container_id"||key=="title"||key=="source") continue
                    if (key=="score") continue;
                    if (key=="data_index") continue;
                    if (key=="prompt_ids" || key =="output_ids") continue;
                    d3.select(`#${clickedWrapperId}`)
                        .select(".expanded-attributed-text-wrapper")
                            .append("div")
                                .attr("class", `expanded-attributed-text-contents`)
                                .attr("id", `expanded-attributed-text-contents-${key}`)
                    d3.select(`#${clickedWrapperId}`)
                        .select(`#expanded-attributed-text-contents-${key}`)
                            .append("span")
                            .attr("class", "expanded-attributed-text-contents-title")
                            .text(key.charAt(0).toUpperCase() + key.slice(1).replace("_", " "))
                    d3.select(`#${clickedWrapperId}`)
                        .select(`#expanded-attributed-text-contents-${key}`)
                            .append("span")
                            .attr("class", "expanded-attributed-text-contents-value")
                            .text(data[key])
                }
                if (data["source"]) {
                    d3.select(`#${clickedWrapperId}`)
                        .select(".expanded-attributed-text-wrapper")
                            .append("div")
                            .attr("class", "expanded-attributed-text-contents")
                            .attr("id", "expanded-attributed-text-contents-source")
                    d3.select(`#${clickedWrapperId}`)
                        .select(`#expanded-attributed-text-contents-source`)
                            .append("span")
                            .attr("class", "expanded-attributed-text-contents-title")
                            .text("Source")
                    d3.select(`#${clickedWrapperId}`)
                        .select(`#expanded-attributed-text-contents-source`)
                            .append("a")
                            .attr("class", "expanded-attributed-text-contents-value")
                            .text(data["title"]?data["title"]:"Link")
                            .attr("href", data["source"])
                }
                else if (data["title"]) {
                    d3.select(`#${clickedWrapperId}`)
                        .select(".expanded-attributed-text-wrapper")
                            .append("div")
                            .attr("class", "expanded-attributed-text-contents")
                            .attr("id", "expanded-attributed-text-contents-source")
                    d3.select(`#${clickedWrapperId}`)
                        .select(`#expanded-attributed-text-contents-title`)
                            .append("span")
                            .attr("class", "expanded-attributed-text-contents-title")
                            .text("Title")
                    d3.select(`#${clickedWrapperId}`)
                        .select(`#expanded-attributed-text-contents-title`)
                            .append("span")
                            .attr("class", "expanded-attributed-text-contents-value")
                            .text(data["title"])
                }
                let expandedContentsHeight = document.querySelector(`#${clickedWrapperId} .expanded-attributed-text-wrapper`).getBoundingClientRect().height
                d3.select(`#${clickedWrapperId}`)
                    .transition()
                    .duration(1000)
                        .style("height", `${expandedContentsHeight+45}px`)
                        .style("pointer", "default")
                        .style("transform", "scale(1)")

                clickedWrapperElement.expanded = true;
                expanded = true;
                expandedAttributedTextWrapperElementId = clickedWrapperId;
                expandedAttributedTextElementId = clickedAttributedTextElementId;

            })
            .on("mouseover", function(event) {
                let targetId = this.id 
                let targetElement = document.getElementById(targetId)
                d3.select(this).style("cursor", targetElement.expanded?"default":"pointer")
                if (!targetElement.expanded) d3.select(`#${targetId}`).style("transform", "scale(1.01)")
            })
            .on("mouseout", function(event) {
                d3.select(this).style("transform", "scale(1)")
            })
    }
}

function setCollapsedText (generated, positive, i, attribution) {
    let textType = generated ? "generated" : "user-provided"
    let positiveType = positive ? "positive" : "negative"
    let elementId = `${textType}-${positiveType}-attributed-text-${i}`
    let element = document.getElementById(elementId)
    element.innerText = ""
    let text = attribution["text"] 

    d3.select(`#${textType}-${positiveType}-attributed-text-info-wrapper-${i}`)
        .append("span")
            .attr("class", `attributed-text-info ${textType}-${positiveType}-attributed-text-info`)
            .text(`#${attribution["data_index"]}`)
    d3.select(`#${textType}-${positiveType}-attributed-text-info-wrapper-${i}`)
        .append("span")
            .attr("class", `attributed-text-info attributed-text-info-score ${textType}-${positiveType}-attributed-text-info`)
            .text(`Score: ${attribution["score"].toFixed(4)}`)

    let word_list = text.split(" ")

    for (word_idx in word_list) {
        let word = word_list[word_idx]
        let original_text = d3.select(`#${element.id}`).text().slice(0,-3)

        if (word_idx==0) d3.select(`#${element.id}`).text(`${word}...`)
        else if (word_idx==word_list.length-1) d3.select(`#${element.id}`).text(`${original_text} ${word}`)
        else d3.select(`#${element.id}`).text(`${original_text} ${word}...`)

        let height = +d3.select(`#${element.id}`).style("height").slice(0,-2)
        let lineheight = +d3.select(`#${element.id}`).style("line-height").slice(0,-2)
        if (height > lineheight*1.5) {
            d3.select(`#${element.id}`).text(`${original_text}...`)
            break
        }
    }
    firstTokenSpacing(d3.select(`#${element.id}`))
}

function collapseExpandedAttributedText(expandedAttributedTextWrapperElementId, expandedAttributedTextElementId) {
    d3.select(`#${expandedAttributedTextWrapperElementId}`)
        .selectAll("div")
        .transition()
        .duration(1000)
            .style("opacity", 0)

    setTimeout(function() {
        d3.select(`#${expandedAttributedTextWrapperElementId}`).html("")
        let attributedTextElementIdSplit = expandedAttributedTextElementId.split("-")
        let generated = attributedTextElementIdSplit[0]=="generated"
        let i = +attributedTextElementIdSplit[attributedTextElementIdSplit.length-1]
        let positive;
        if (generated) {positive = attributedTextElementIdSplit[1]=="positive"}
        else {positive = attributedTextElementIdSplit[2]=="positive"}

        let generatedType = generated ? "generated" : "user-provided"
        let positiveType = positive ? "positive" : "negative"

        d3.select(`#${expandedAttributedTextWrapperElementId}`)
            .append("div")
            .attr("class", `attributed-text-info-wrapper ${generatedType}-${positiveType}-attributed-text-info-wrapper`)
            .attr("id", `${generatedType}-${positiveType}-attributed-text-info-wrapper-${i}`)
        d3.select(`#${expandedAttributedTextWrapperElementId}`)
            .append("div")
            .attr("class", `attributed-text ${generatedType}-${positiveType}-attributed-text`)
            .attr("id", expandedAttributedTextElementId)
        let attributedTextElement = document.getElementById(expandedAttributedTextElementId)
        let attribution = attributedTextElement.parentElement.attributionData
        setCollapsedText(generated, positive, i, attribution)
        document.getElementById(expandedAttributedTextWrapperElementId).expanded = false
    }, 1000)

    d3.select(`#${expandedAttributedTextWrapperElementId}`)
        .transition()
        .duration(1000)
            .style("height", "54px")
}

function firstTokenSpacing (tokensContainer) {
    tokensContainer.selectAll(".token")
        .each(function() {
            let offsetLeft = document.getElementById(this.id).offsetLeft

            if (offsetLeft < 1) this.classList.add("first-in-line-token")
            else this.classList.remove("first-in-line-token")
        })
}

function displayTFIDF (generated, positive) {
    let textType = generated ? "generated" : "user-provided"
    let positiveType = positive ? "positive" : "negative"
    let color = generated ? "blue":"red"
    let tfidf = positive ? (generated ? gPosTfIdf : uPosTfIdf) : (generated ? gNegTfIdf : uNegTfIdf)
    let num_shown = positive ? (generated?num_g_pos_shown:num_u_pos_shown) : (generated?num_g_neg_shown:num_u_neg_shown)
    if (num_shown == 0) {
        d3.select(`.${textType}-${positiveType}-attribution-text-tf-idf`)
            .html("")
            .style("width", `calc(100% - 5px - ${(+d3.select(".attribution-text-tf-idf-title-wrapper").style("width").slice(0,-2))}px)`)
            .style("vertical-align", "top")
            .style("margin-top", "12px")
            .text("No data to display")
        return;
    }

    d3.select(`.${textType}-${positiveType}-attribution-text-tf-idf`)
        .html("")
        .style("margin-top", "")
        .style("width", `calc(100% - 5px - ${(+d3.select(".attribution-text-tf-idf-title-wrapper").style("width").slice(0,-2))}px)`)
        .style("float", "right")
        .style("white-space", "nowrap")
        .style("scrollbar-width", "none")
        .style("overflow-x", "scroll")
        .style("overflow-y", "hidden")
        .style("height", "35px")
        .style("position", "relative")
        .on("mouseover", function(event) {  // Mouseover Event Listener
            d3.select(this).style("scrollbar-width", "thin")
        })
        .on("mouseout", function(event) {  // Mouseover Event Listener
            d3.select(this).style("scrollbar-width", "none")
        })
        
    d3.select(`.${textType}-${positiveType}-attribution-text-tf-idf`)
        .append("div")
        .attr("class", `attribution-text-tf-idf-dummy-word`)
        .style("width", "10px")
        .style("display", "inline-block")
    d3.select(`.${textType}-${positiveType}-attribution-text-tf-idf`)
        .selectAll(`.${textType}-${positiveType}-attribution-text-tf-idf-token`)
        .data(tfidf[num_shown-1])
        .enter()
        .append("div")
            .attr("class", `attribution-text-tf-idf-token ${textType}-${positiveType}-attribution-text-tf-idf-token`)
            .style("background-color", `var(--${color}${positive?1:0})`)
            .style("color", `var(--${color}${positive?5:4})`)
            .style("border", `1px solid var(--${color}${positive?3:2})`)
            .text(d => d[0])
            .on("mouseover", function(event) {  // Mouseover Event Listener
                let data = d3.select(this).data()[0]
                if (generated) {
                    d3.select(this)
                        .style("background-color", `var(--${color}${positive?2:1})`)
                        .style("border", `2px solid var(--${color}${positive?6:4})`)
                        .style("padding", "1.5px 3.5px")
                }
                else {
                    d3.select(this)
                        .style("background-color", `var(--${color}${positive?2:1})`)
                        .style("border", `2px solid var(--${color}${positive?5:4}`)
                        .style("padding", "1.5px 3.5px")
                }
                for(let i=0; i<num_shown; i++) {
                    if (data[2].includes(i)) {
                        if (generated) {
                            d3.select(`#${textType}-${positiveType}-attributed-text-wrapper-${i}`)
                                .style("border", `3px solid var(--${color}${positive?6:4})`)
                                .style("padding", `5.5px 10px`)
                        }
                        else {
                            d3.select(`#${textType}-${positiveType}-attributed-text-wrapper-${i}`)
                                .style("border", `3px solid var(--${color}${positive?5:4})`)
                                .style("padding", `5.5px 10px`)
                        }
                    }
                }
            })
            .on("mouseout", function(event) {  // Mouseover Event Listener
                d3.select(this)
                    .style("background-color", `var(--${color}${positive?1:0})`)
                    .style("padding", "2.5px 4.5px")
                    .style("border", `1px solid var(--${color}${positive?3:2})`)
                d3.selectAll(`.${textType}-${positiveType}-attributed-text-wrapper`)
                    .style("border", `1px solid var(--${color}${positive?4:3})`)
                    .style("padding", `7.5px 12px`)
            })
    d3.select(`.${textType}-${positiveType}-attribution-text-tf-idf`)
        .append("div")
        .attr("class", `attribution-text-tf-idf-dummy-word`)
        .style("width", "25px")
        .style("display", "inline-block")

    d3.select(`.${textType}-${positiveType}-attribution-text-tf-idf-wrapper`)
        .append("div")
        .attr("class", `attribution-text-tf-idf-gradient`)
        .style("width", "25px")
        .style("height", "35px")
        .style("position", "absolute")
        .style("top", "0")
        .style("right", "0")
        .style("z-index", "10")
        .style("background", `linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,1))`)
        .style("pointer-events", "none")
    d3.select(`.${textType}-${positiveType}-attribution-text-tf-idf-wrapper`)
        .append("div")
        .attr("class", `attribution-text-tf-idf-gradient`)
        .style("width", "25px")
        .style("height", "35px")
        .style("position", "absolute")
        .style("top", "0")
        .style("left", "60px")
        .style("z-index", "10")
        .style("background", `linear-gradient(to left, rgba(255,255,255,0), rgba(255,255,255,1))`)
        .style("pointer-events", "none")
}

function drawHistogram() {
    let margin = {top: 5, right: 40, bottom: 5, left: 40}
    let parentWidth = document.getElementsByClassName("attribution-score-histogram")[0].getBoundingClientRect().width
    let svgWidth = 340
    let width = svgWidth - margin.left - margin.right
    let height = 110 - margin.top - margin.bottom;
    let labelTextHeight = 10;

    let lineColor = "#ffffff";

    let svg = d3.select(".attribution-score-histogram").append("svg")
        .attr("width", width+margin.left+margin.right)
        .attr("height", height+margin.top+margin.bottom)
        .style("margin-left", `${(parentWidth-svgWidth)/2}px`)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`)

    let y = d3.scaleLinear().domain([1, -1]).range([0, height])

    
    // generated
    let x = d3.scaleSqrt().range([0, width/2]).domain([0, Math.max(d3.max(gScoreHistogramCounts, d => d[2]), d3.max(uScoreHistogramCounts, d => d[2]))])
    svg.append("g")
        .attr("class", "generated-attribution-score-histogram-bars-wrapper")
        .selectAll("rect")
        .data(gScoreHistogramCounts)
        .enter()
        .append("rect")
        .attr("class", (d,i) => "attribution-score-histogram-bars generated-attribution-score-histogram-bars generated-attribution-score-histogram-bar-"+i)
        .attr("y", 0)
        .attr("width", d => x(d[2]))
        .attr("height", d => y(d[0])-y(d[1]))
        .attr("transform", d=>`translate(${width/2-x(d[2])},${y(d[1])})`)
        .attr("fill", "var(--blue1)")
        .attr("stroke", "#ffffff")

    // user-provided
    svg.append("g")
        .attr("class", "user-provided-attribution-score-histogram-bars-wrapper")
        .selectAll("rect")
        .data(uScoreHistogramCounts)
        .enter()
        .append("rect")
        .attr("class", (d,i) => "attribution-score-histogram-bars user-provided-attribution-score-histogram-bars user-provided-attribution-score-histogram-bar-"+i)
        .attr("y", 0)
        .attr("width", d => x(d[2]))
        .attr("height", d => y(d[0])-y(d[1]))
        .attr("transform", d=>`translate(${width/2},${y(d[1])})`)
        .attr("fill", "var(--red1)")
        .attr("stroke", "#ffffff")

    // tick labels
    svg.append("g")
        .attr("class", "attribution-score-histogram-y-axis")
        .attr("transform", `translate(${width/2},0)`)
        .append("path")
            .attr("d", `M 0 0 L 0 ${height}`)
            .attr("stroke", "#e0e0e0")
            .attr("stroke-width", 1)
            .attr("class", "attribution-score-histogram-y-axis-line")
    d3.select(".attribution-score-histogram-y-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-y-axis-labels attribution-score-histogram-y-axis-label-max")
            .append("text")
            .attr("class", "attribution-score-histogram-axis-label")
                .text("1")
                .style("font-size", "12px")
                .style("fill", "#a0a0a0")
                .style("text-anchor", "end")
                .attr("x", -4)
                .attr("y", 7)
    d3.select(".attribution-score-histogram-y-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-y-axis-labels attribution-score-histogram-y-axis-label-min")
            .append("text")
            .attr("class", "attribution-score-histogram-axis-label")
                .text("-1")
                .style("font-size", "12px")
                .style("fill", "#a0a0a0")
                .style("text-anchor", "end")
                .attr("x", -4)
                .attr("y", height+labelTextHeight-7)
    svg.append("g")
        .attr("class", "attribution-score-histogram-x-axis")
        .attr("transform", `translate(${width/2},${height/2})`)
        .append("path")
            .attr("d", `M 0 0 L 5 0`)
            .attr("stroke", "#e0e0e0")
            .attr("stroke-width", 1)
            .attr("class", "attribution-score-histogram-x-axis-line")
    d3.select(".attribution-score-histogram-x-axis")
        .append("text")
            .attr("class", "attribution-score-histogram-axis-label")
            .text("0")
            .style("font-size", "12px")
            .style("fill", "#c0c0c0")
            .style("text-anchor", "end")
            .attr("x", -4)
            .attr("y", 5)

    // hovered
    svg.append("text")
        .attr("class", "attribution-score-histogram-hovered-text")
        .style("font-size", "12px")
    
    svg.append("g")
        .attr("class", "generated-attribution-score-histogram-bars-hovered-wrapper")
        .selectAll("rect")
        .data(gScoreHistogramCounts)
        .enter()
        .append("rect")
        .attr("class", (d,i) => "attribution-score-histogram-bars-hovered generated-attribution-score-histogram-bars-hovered generated-attribution-score-histogram-bar-hovered-"+i)
        .attr("y", 0)
        .attr("width", d => width/2)
        .attr("height", d => y(d[0])-y(d[1]))
        .attr("transform", d=>`translate(0,${y(d[1])})`)
        .attr("fill", "#ffffff00")
        .on("mouseover", function (event) {
            let bar_index = +this.classList[2].split("-").pop()
            let bin = d3.select(this).data()[0]
            d3.select(`.generated-attribution-score-histogram-bar-${bar_index}`).style("fill", "var(--blue3)")
            d3.selectAll(".attribution-score-histogram-axis-label").style("opacity", "0")
            svg.select(".attribution-score-histogram-hovered-text")
                .attr("text-anchor", "end")
                .style("fill", "var(--blue5)")
                .attr("x", width/2-x(bin[2])-5)
                .attr("y", y(bin[0])-1)
                .text(bin[2])

            for (let i=0 ; i<num_g_pos_shown ; i++) {
                let attributedTextWrapperElement = document.getElementById(`generated-positive-attributed-text-wrapper-${i}`)
                let data = attributedTextWrapperElement.attributionData
                if (data["score_histogram_bin"] == bar_index) {
                    d3.select(`#generated-positive-attributed-text-wrapper-${i}`)
                        .style("border", `3px solid var(--blue6)`)
                        .style("padding", `5.5px 10px`)
                    histogramHighlightedTextWrapperElementIds.push(`generated-positive-attributed-text-wrapper-${i}`)
                }
            }

            for (let i=0 ; i<num_g_neg_shown ; i++) {
                let attributedTextWrapperElement = document.getElementById(`generated-negative-attributed-text-wrapper-${i}`)
                let data = attributedTextWrapperElement.attributionData
                if (data["score_histogram_bin"] == bar_index) {
                    d3.select(`#generated-negative-attributed-text-wrapper-${i}`)
                        .style("border", `3px solid var(--blue4)`)
                        .style("padding", `5.5px 10px`)
                    histogramHighlightedTextWrapperElementIds.push(`generated-negative-attributed-text-wrapper-${i}`)
                }
            }
        })
        .on("mouseout", function (event) {
            let bar_index = +this.classList[2].split("-").pop()
            d3.select(`.generated-attribution-score-histogram-bar-${bar_index}`).style("fill", "var(--blue1)")
            d3.select(".attribution-score-histogram-hovered-text").text("")
            d3.selectAll(".attribution-score-histogram-axis-label").style("opacity", "1")
            for (let i=0 ; i<histogramHighlightedTextWrapperElementIds.length ; i++) {
                let positive = histogramHighlightedTextWrapperElementIds[i].split("-")[1]=="positive"
                d3.select(`#${histogramHighlightedTextWrapperElementIds[i]}`)
                    .style("border", `1px solid var(--blue${positive?4:3})`)
                    .style("padding", `7.5px 12px`)
            }
            histogramHighlightedTextWrapperElementIds = []
        })
    
    // user-provided
    svg.append("g")
        .attr("class", "generated-attribution-score-histogram-bars-hovered-wrapper")
        .selectAll("rect")
        .data(uScoreHistogramCounts)
        .enter()
        .append("rect")
        .attr("class", (d,i) => "attribution-score-histogram-bars-hovered user-provided-attribution-score-histogram-bars-hovered user-provided-attribution-score-histogram-bar-hovered-"+i)
        .attr("y", 0)
        .attr("width", d => width/2)
        .attr("height", d => y(d[0])-y(d[1]))
        .attr("transform", d=>`translate(${width/2},${y(d[1])})`)
        .attr("fill", "#ffffff00")
        .on("mouseover", function (event) {
            let bar_index = +this.classList[2].split("-").pop()
            let bin = d3.select(this).data()[0]
            d3.select(`.user-provided-attribution-score-histogram-bar-${bar_index}`).style("fill", "var(--red2)")
            d3.selectAll(".attribution-score-histogram-axis-label").style("opacity", "0")

            svg.select(".attribution-score-histogram-hovered-text")
                .attr("text-anchor", "start")
                .style("fill", "var(--red5)")
                .style("font-size", "12px")
                .style("pointer-event", "none")
                .attr("x", width/2+x(bin[2])+5)
                .attr("y", y(bin[0])-1)
                .text(bin[2])

            for (let i=0 ; i<num_u_pos_shown ; i++) {
                let attributedTextWrapperElement = document.getElementById(`user-provided-positive-attributed-text-wrapper-${i}`)
                let data = attributedTextWrapperElement.attributionData
                if (data["score_histogram_bin"] == bar_index) {
                    d3.select(`#user-provided-positive-attributed-text-wrapper-${i}`)
                        .style("border", `3px solid var(--red5)`)
                        .style("padding", `5.5px 10px`)
                    histogramHighlightedTextWrapperElementIds.push(`user-provided-positive-attributed-text-wrapper-${i}`)
                }
            }

            for (let i=0 ; i<num_u_neg_shown ; i++) {
                let attributedTextWrapperElement = document.getElementById(`user-provided-negative-attributed-text-wrapper-${i}`)
                let data = attributedTextWrapperElement.attributionData
                if (data["score_histogram_bin"] == bar_index) {
                    d3.select(`#user-provided-negative-attributed-text-wrapper-${i}`)
                        .style("border", `3px solid var(--red4)`)
                        .style("padding", `5.5px 10px`)
                    histogramHighlightedTextWrapperElementIds.push(`user-provided-negative-attributed-text-wrapper-${i}`)
                }
            }
        })
        .on("mouseout", function (event) {
            let bar_index = +this.classList[2].split("-").pop()
            d3.select(`.user-provided-attribution-score-histogram-bar-${bar_index}`).style("fill", "var(--red1)")
            d3.select(".attribution-score-histogram-hovered-text").text("")
            d3.selectAll(".attribution-score-histogram-axis-label").style("opacity", "1")
            for (let i=0 ; i<histogramHighlightedTextWrapperElementIds.length ; i++) {
                let positive = histogramHighlightedTextWrapperElementIds[i].split("-")[2]=="positive"
                let color = histogramHighlightedTextWrapperElementIds[i].split("-")[2]=="positive"?"var(--red1)":"var(--red0)"
                d3.select(`#${histogramHighlightedTextWrapperElementIds[i]}`)
                    .style("border", `1px solid var(--red${positive?4:3})`)
                    .style("padding", `7.5px 12px`)
            }
            histogramHighlightedTextWrapperElementIds = []
        })
}

