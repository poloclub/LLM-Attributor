let positiveAttribution, negativeAttribution
let positiveMaxNum, negativeMaxNum;
let num_pos_shown = 3
let num_neg_shown = 0
let posTfIdf, negTfIdf

let expanded = false;
let expandedAttributedTextWrapperElementId = "";
let expandedAttributedTextElementId = "";
let histogramHighlightedTextWrapperElementIds = [];

document.addEventListener("attribute", function(event) {
    console.log(event)
    positiveAttribution = event.positiveAttribution
    negativeAttribution = event.negativeAttribution
    positiveMaxNum = event.positiveMaxNum
    negativeMaxNum = event.negativeMaxNum
    posTfIdf = event.posTfIdf
    negTfIdf = event.negTfIdf

    // positive dropdown
    d3.select(".positive-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-title-wrapper positive-attribution-num-dropdown-title-wrapper")
            .on("mouseover", function(event) {  // Mouseover Event Listener
                d3.select(".positive-attribution-num-dropdown-svg").style("fill", "#313695f0")
                d3.select(".positive-attribution-num-dropdown-text").style("fill", "#313695f0")
            })
            .on("mouseout", function(event) {  // Mouseover Event Listener
                d3.select(".positive-attribution-num-dropdown-svg").style("fill", "#313695b0")
                d3.select(".positive-attribution-num-dropdown-text").style("fill", "#313695b0")
            })
            .on("click", function(event) {  // Click Event Listener
                d3.select(".positive-attribution-num-dropdown-options-wrapper")
                    .style("display", d3.select(".positive-attribution-num-dropdown-options-wrapper").style("display")=="none"?"block":"none")  
                document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0].expanded = !document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0].expanded
            })
            .append("div")
                .attr("class", "attribution-num-dropdown-text positive-attribution-num-dropdown-text")
                .text(num_pos_shown)
    document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0].expanded = false
    d3.select(".positive-attribution-num-dropdown-title-wrapper")
        .append("svg")
            .attr("class", "attribution-num-dropdown-svg positive-attribution-num-dropdown-svg")
            .append("g")
            .append("path")
            .attr("d", "M 0 0 L 10 0 L 5 10 Z")
    d3.select(".positive-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-options-wrapper positive-attribution-num-dropdown-options-wrapper")
            .selectAll("p")
            .data([...Array(positiveMaxNum+1).keys()])
            .enter()
            .append("p")
                .attr("class", "attribution-num-dropdown-option positive-attribution-num-dropdown-option")
                .text(d=>d)
                .on("mouseover", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#e0e0e0")
                })
                .on("mouseout", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#f0f0f0")
                })
                .on("click", function(event) {  // Click Event Listener
                    num_pos_shown = +d3.select(this).text()
                    d3.select(".positive-attribution-num-dropdown-text").text(num_pos_shown)
                    displayAttribution(positiveAttribution, positive=true)
                    summarizeAttributedData(posTfIdf, true)
                    adjustSidebarHeight()
                    d3.select(".positive-attribution-num-dropdown-options-wrapper").style("display", "none")
                })

    // negative dropdown
    d3.select(".negative-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-title-wrapper negative-attribution-num-dropdown-title-wrapper")
            .on("mouseover", function(event) {  // Mouseover Event Listener
                d3.select(".negative-attribution-num-dropdown-svg").style("fill", "#a50026f0")
                d3.select(".negative-attribution-num-dropdown-text").style("fill", "#a50026f0")
            })
            .on("mouseout", function(event) {  // Mouseover Event Listener
                d3.select(".negative-attribution-num-dropdown-svg").style("fill", "#a50026b0")
                d3.select(".negative-attribution-num-dropdown-text").style("fill", "#a50026b0")
            })
            .on("click", function(event) {  // Click Event Listener
                d3.select(".negative-attribution-num-dropdown-options-wrapper")
                    .style("display", d3.select(".negative-attribution-num-dropdown-options-wrapper").style("display")=="none"?"block":"none")  
                document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0].expanded = !document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0].expanded
            })
            .append("div")
                .attr("class", "attribution-num-dropdown-text negative-attribution-num-dropdown-text")
                .text(num_neg_shown)
    document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0].expanded = false
    d3.select(".negative-attribution-num-dropdown-title-wrapper")
        .append("svg")
            .attr("class", "attribution-num-dropdown-svg negative-attribution-num-dropdown-svg")
            .append("g")
            .append("path")
            .attr("d", "M 0 0 L 10 0 L 5 10 Z")
    d3.select(".negative-attribution-num-dropdown-wrapper")
        .append("div")
            .attr("class", "attribution-num-dropdown-options-wrapper negative-attribution-num-dropdown-options-wrapper")
            .selectAll("p")
            .data([...Array(negativeMaxNum+1).keys()])
            .enter()
            .append("p")
                .attr("class", "attribution-num-dropdown-option negative-attribution-num-dropdown-option")
                .text(d=>d)
                .on("mouseover", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#e0e0e0")
                })
                .on("mouseout", function(event) {  // Mouseover Event Listener
                    d3.select(this).style("background-color", "#f0f0f0")
                })
                .on("click", function(event) {  // Click Event Listener
                    num_neg_shown = +d3.select(this).text()
                    d3.select(".negative-attribution-num-dropdown-text").text(num_neg_shown)
                    displayAttribution(negativeAttribution, positive=false)
                    summarizeAttributedData(negTfIdf, false)
                    adjustSidebarHeight()
                    d3.select(".negative-attribution-num-dropdown-options-wrapper").style("display", "none")
                })

    displayAttribution(positiveAttribution, positive=true)
    displayAttribution(negativeAttribution, positive=false)

    drawHistogram(event.scoreHistogramCounts)
    resetAttributedDataSummary(posTfIdf, negTfIdf)
})

document.addEventListener("resize", function(event) {
    // TODO
})

document.addEventListener("mouseup", function(event) {
    // if click is in dropdown box, do nothing
    let posDropdownWrapper = document.getElementsByClassName("positive-attribution-num-dropdown-wrapper")[0]
    let negDropdownWrapper = document.getElementsByClassName("negative-attribution-num-dropdown-wrapper")[0]
    let posDropdownBox = posDropdownWrapper.getBoundingClientRect()
    let negDropdownBox = negDropdownWrapper.getBoundingClientRect()
    let posDropdownBoxLeft = posDropdownBox.x, posDropdownBoxRight = posDropdownBox.x + posDropdownBox.width, posDropdownBoxTop = posDropdownBox.y, posDropdownBoxBottom = posDropdownBox.y + posDropdownBox.height
    let negDropdownBoxLeft = negDropdownBox.x, negDropdownBoxRight = negDropdownBox.x + negDropdownBox.width, negDropdownBoxTop = negDropdownBox.y, negDropdownBoxBottom = negDropdownBox.y + negDropdownBox.height

    if (event.clientX > posDropdownBoxLeft && event.clientX < posDropdownBoxRight && event.clientY > posDropdownBoxTop && event.clientY < posDropdownBoxBottom) return;
    if (event.clientX > negDropdownBoxLeft && event.clientX < negDropdownBoxRight && event.clientY > negDropdownBoxTop && event.clientY < negDropdownBoxBottom) return;

    // if dropdown is expanded, collapse it
    if (posDropdownWrapper.expanded) {
        d3.select(".positive-attribution-num-dropdown-options-wrapper").style("display", "none")
        posDropdownWrapper.expanded = false
    }  
    if (negDropdownWrapper.expanded) {
        d3.select(".negative-attribution-num-dropdown-options-wrapper").style("display", "none")
        negDropdownWrapper.expanded = false
    }  

    // if any attribution box is expanded and click is outside the box, collapse it (be careful the conflict between the original collapse)
    if (expanded) {
        let expandedBox = document.getElementById(expandedAttributedTextWrapperElementId).getBoundingClientRect()
        let expandedBoxLeft = expandedBox.x, expandedBoxRight = expandedBox.x + expandedBox.width, expandedBoxTop = expandedBox.y, expandedBoxBottom = expandedBox.y + expandedBox.height
        if (event.clientX > expandedBoxLeft && event.clientX < expandedBoxRight && event.clientY > expandedBoxTop && event.clientY < expandedBoxBottom) {}
        else {
            collapseExpandedAttributedText(expandedAttributedTextWrapperElementId, expandedAttributedTextElementId);
            // TODO: Change expanded if the clicked area is not on any attribution box
            if (!event.target.classList.contains("attributed-text-wrapper")) {
                expanded = false;
                expandedAttributedTextWrapperElementId = "";
                expandedAttributedTextElementId = "";
            }
        }
    }
})

function firstTokenSpacing (tokensContainer) {
    tokensContainer.selectAll(".token")
        .each(function() {
            let offsetLeft = document.getElementById(this.id).offsetLeft

            if (offsetLeft < 19) this.classList.add("first-in-line-token")
            else this.classList.remove("first-in-line-token")
        })
}

function displayAttribution (attribution, positive=true) {
    let type = positive?"positive":"negative"
    let num_shown = positive?num_pos_shown:num_neg_shown
    d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").remove()

    for (let i=0; i<num_shown; i++) {
        if (attribution.length <= i) break;
        d3.select(`.${type}-attribution`)
            .append("div")
                .attr("class", `attributed-text-wrapper ${type}-attributed-text-wrapper`)
                .attr("id", `${type}-attributed-text-wrapper-${i}`)
        
        let attributedTextWrapperElement = document.getElementById(`${type}-attributed-text-wrapper-${i}`)
        attributedTextWrapperElement.attributionData = attribution[i]
        attributedTextWrapperElement.positive = positive

        d3.select(`#${type}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("id", `${type}-attributed-text-info-wrapper-${i}`)
                .attr("class", `attributed-text-info-wrapper ${type}-attributed-text-info-wrapper`)
        d3.select(`#${type}-attributed-text-wrapper-${i}`)
            .append("div")
                .attr("class", `attributed-text ${type}-attributed-text`)
                .attr("id", `${type}-attributed-text-${i}`)
        let attributedTextElement = document.getElementById(`${type}-attributed-text-${i}`)
        setCollapsedText(attribution[i], attributedTextElement)
                

        // Click Event Listener
        d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").on("click", function(event) {
            // tokens-container: for the expanded
            // attributed-text: for the collapsed
            let clickedWrapperId = event.target.id
            console.log(clickedWrapperId)
            let clickedWrapperElement = document.getElementById(clickedWrapperId)  // wrapper
            if (expanded && clickedWrapperElement.expanded) return;

            let clickedAttributedTextElementId = d3.select(`#${clickedWrapperId}`).select(".attributed-text").attr("id")
            let clickedAttributedTextElement = document.getElementById(clickedAttributedTextElementId)

            let data = clickedWrapperElement.attributionData
            d3.select(`#${clickedWrapperId}`).html("")

            // Expand
            // TODO: Change order; first add contents with opacity 0, and expand to the size that fits to the added contents
            d3.select(`#${clickedWrapperId}`)
                .append("div")
                    .attr("class", "expanded-contents-wrapper")
                    .attr("id", "expanded-contents-wrapper")
                    // .style("opacity", 0)
            d3.select(`#${clickedWrapperId}`)
                .select("#expanded-contents-wrapper")
                .append("div")
                    .attr("class", "attributed-text-expanded-title")
                    .text("Full text")
            d3.select(`#${clickedWrapperId}`)
                .select(".attributed-text-expanded-title")
                .append("span")
                .style("padding-left", "5px")
                .style("font-weight", "100")
                .style("font-size", "12px")
                .style("color", "#a0a0a0")
                .text("Only black is in the data; gray has been added for better understanding of context")
            d3.select(`#${clickedWrapperId}`)
                .select("#expanded-contents-wrapper")
                .append("div")
                .attr("class", "full-text-wrapper")
                .html(data["text_html_code"])
            let tokensContainerId = d3.select(`#${clickedWrapperId}`).select(".tokens-container").attr("id")
            firstTokenSpacing(d3.select(`#${tokensContainerId}`))

            for (let key in data) {
                if (key=="text_html_code"||key=="text"||key=="score_histogram_bin"||key=="tokens_container_id"||key=="title"||key=="source") continue
                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                        .append("div")
                            .attr("class", `expanded-contents`)
                            .attr("id", `expanded-contents-${key}`)
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-${key}`)
                        .append("span")
                        .attr("class", "expanded-contents-title")
                        .text(key=="score"?"Attribution score":key.charAt(0).toUpperCase() + key.slice(1).replace("_", " "))
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-${key}`)
                        .append("span")
                        .attr("class", "expanded-contents-value")
                        .text(data[key])
            }
            if (data["source"]) {
                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                        .append("div")
                        .attr("class", "expanded-contents")
                        .attr("id", "expanded-contents-source")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-source`)
                        .append("span")
                        .attr("class", "expanded-contents-title")
                        .text("Source")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-source`)
                        .append("a")
                        .attr("class", "expanded-contents-value")
                        .text(data["title"]?data["title"]:"Link")
                        .attr("href", data["source"])
            }
            else if (data["title"]) {
                d3.select(`#${clickedWrapperId}`)
                    .select("#expanded-contents-wrapper")
                        .append("div")
                        .attr("class", "expanded-contents")
                        .attr("id", "expanded-contents-source")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-title`)
                        .append("span")
                        .attr("class", "expanded-contents-title")
                        .text("Title")
                d3.select(`#${clickedWrapperId}`)
                    .select(`#expanded-contents-title`)
                        .append("span")
                        .attr("class", "expanded-contents-value")
                        .text(data["title"])
            }


            // let expandedContentsHeight = document.getElementById("expanded-contents-wrapper").getBoundingClientRect().height
            let expandedContentsHeight = document.querySelector(`#${clickedWrapperId} #expanded-contents-wrapper`).getBoundingClientRect().height
            d3.select(`#${clickedWrapperId}`)
                .transition()
                .duration(500)
                    .style("height", `${expandedContentsHeight+27.015}px`)
                    .style("pointer", "default")
                    .style("transform", "scale(1)")

            clickedWrapperElement.expanded = true;
            expanded = true;
            expandedAttributedTextWrapperElementId = clickedWrapperId;
            expandedAttributedTextElementId = clickedAttributedTextElementId;

        })

        d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").on("mouseover", function(event) {
            let targetElement = event.target
            let targetId = targetElement.id 
            d3.select(`#${targetId}`).style("cursor", targetElement.expanded?"default":"pointer")

            if (!targetElement.expanded) d3.select(`#${targetId}`).style("transform", "scale(1.01)")
        })

        d3.select(`.${type}-attribution`).selectAll(".attributed-text-wrapper").on("mouseout", function(event) {
            d3.select(`#${event.target.id}`).style("transform", "scale(1)")
        })
    }
}

function collapseExpandedAttributedText (attributedTextWrapperElementId, attributedTextElementId) {
    d3.select(`#${attributedTextWrapperElementId}`)
        .selectAll("div")
            .transition()
            .duration(500)
                .style("opacity", "0")
    console.log("collapse; attributedTextWrapperElementId", attributedTextWrapperElementId)
    setTimeout(function() {
        // rewrite the collapsed text box
        d3.select(`#${attributedTextWrapperElementId}`).html("")
        let attributedTextElementIdSplit = attributedTextElementId.split("-")
        let type = attributedTextElementIdSplit[0]
        let i = attributedTextElementIdSplit[attributedTextElementIdSplit.length-1]
        d3.select(`#${attributedTextWrapperElementId}`)
            .append("div")
                .attr("id", `${type}-attributed-text-info-wrapper-${i}`)
                .attr("class", `attributed-text-info-wrapper ${type}-attributed-text-info-wrapper`)
        d3.select(`#${attributedTextWrapperElementId}`)
            .append("div")
                .attr("class", `attributed-text ${type}-attributed-text`)
                .attr("id", attributedTextElementId)
        let expandedAttributedTextElement = document.getElementById(attributedTextElementId)
        let attribution = document.getElementById(attributedTextWrapperElementId).attributionData
        setCollapsedText(attribution, expandedAttributedTextElement)
        document.getElementById(attributedTextWrapperElementId).expanded = false;
    }, 500)

    // Collapse the expanded wrapper (size)
    d3.select(`#${attributedTextWrapperElementId}`)
        .transition()
        .duration(500)
            .style("height", "54px")
}

function setCollapsedText (attribution, element) {
    element.innerText = ""
    let elementId = element.id;
    let elementIdSplit = elementId.split("-")
    let type = elementIdSplit[0]
    let i = elementIdSplit[elementIdSplit.length-1]
    let text = attribution["text"] 

    d3.select(`#${type}-attributed-text-info-wrapper-${i}`)
        .append("span")
            .attr("class", `attributed-text-info ${type}-attributed-text-info`)
            .html(`Training Data #${attribution["data_index"]}`)
    d3.select(`#${type}-attributed-text-info-wrapper-${i}`)
        .append("span")
            .attr("class", `attributed-text-info ${type}-attributed-text-info`)
            .html(`<i class="fa-regular fa-star"></i>Score: ${attribution["score"].toExponential()}`)
            // .html(`Score: ${attribution[i]["score"].toExponential()}`)
    
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

function drawHistogram(counts) {
    // draw histogram of attribution scores and highlight the bars when mouseover the attribution box
    let margin = {top: 20, right: 20, bottom: 0, left: 20}
    let parentWidth = document.getElementsByClassName("attribution-score-histogram")[0].getBoundingClientRect().width

    let width = parentWidth - margin.left - margin.right
    let height = 100 - margin.top - margin.bottom;

    let lineColor = "#d0d0d0";
    let fillColor = "#d0d0d0";

    let svg = d3.select(".attribution-score-histogram").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom+25)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let x = d3.scaleLinear()
        .domain([Math.min(...counts.map(d=>d[0])), Math.max(...counts.map(d=>d[1]))])
        .range([0, width])
    svg.append("g")
        .attr("class", "attribution-score-histogram-x-axis")
        .attr("transform", "translate(0," + height + ")")
        .append("path")
            .attr("class", "attribution-score-histogram-x-axis-line")
            .attr("d", `M 0 0 L ${width} 0`)
            .attr("stroke", lineColor)
    d3.select(".attribution-score-histogram-x-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-x-axis-ticks attribution-score-histogram-x-axis-ticks-min")
            .append("path")
                .attr("class", "attribution-score-histogram-x-axis-tick")
                .attr("d", `M 0 0 L 0 5`)
                .attr("stroke", lineColor)
    d3.select(".attribution-score-histogram-x-axis-ticks-min")
        .append("text")
            .attr("class", "attribution-score-histogram-x-axis-tick-text attribution-score-histogram-x-axis-tick-text-min")
            .text(counts[0][0].toExponential())
    let minTextWidth = d3.select(".attribution-score-histogram-x-axis-tick-text-min").node().getBBox().width
    d3.select(".attribution-score-histogram-x-axis-tick-text-min")
        .attr("x", -minTextWidth/2)
        .attr("y", 16)
    d3.select(".attribution-score-histogram-x-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-x-axis-ticks attribution-score-histogram-x-axis-ticks-max")
            .append("path")
                .attr("class", "attribution-score-histogram-x-axis-tick")
                .attr("d", `M ${width} 0 L ${width} 5`)
                .attr("stroke", lineColor)
    d3.select(".attribution-score-histogram-x-axis-ticks-max")
        .append("text")
            .attr("class", "attribution-score-histogram-x-axis-tick-text attribution-score-histogram-x-axis-tick-text-max")
            .text(counts[counts.length-1][1].toExponential())
    let maxTextWidth = d3.select(".attribution-score-histogram-x-axis-tick-text-max").node().getBBox().width
    d3.select(".attribution-score-histogram-x-axis-tick-text-max")
        .attr("x", width-maxTextWidth/2)
        .attr("y", 16)
    d3.select(".attribution-score-histogram-x-axis")
        .append("g")
            .attr("class", "attribution-score-histogram-x-axis-ticks attribution-score-histogram-x-axis-ticks-zero")
            .append("path")
                .attr("class", "attribution-score-histogram-x-axis-tick")
                .attr("d", `M ${x(0)} 0 L ${x(0)} 5`)
                .attr("stroke", lineColor)
    d3.select(".attribution-score-histogram-x-axis-ticks-zero")
        .append("text")
            .attr("class", "attribution-score-histogram-x-axis-tick-text attribution-score-histogram-x-axis-tick-text-zero")
            .text(0)
    let zeroTextWidth = d3.select(".attribution-score-histogram-x-axis-tick-text-zero").node().getBBox().width
    d3.select(".attribution-score-histogram-x-axis-tick-text-zero")
        .attr("x", x(0)-zeroTextWidth/2)
        .attr("y", 16)
    
    


    // let y = d3.scaleLinear()
    let y = d3.scaleSqrt()
        .range([height,0])
    y.domain([0, Math.max(...counts.map(d=>d[2]))])
    // let y = d3.scaleLog()
        // .range([height,0])
    // y.domain([1, Math.max(...counts.map(d=>d[2]))])
    // svg.append("g").call(d3.axisLeft(y))
    // TODO: Add y-axis and ticks

    svg.append("g")
        .attr("class", "attribution-score-histogram-bars-wrapper")
        .selectAll("rect")
        .data(counts)
        .enter()
        .append("rect")
            .attr("x", 0)
            .attr("transform", d => {
                return `translate(${x(d[0])}, ${y(d[2])})`})
            .attr("width", d => x(d[1]) - x(d[0]) - 0)
            .attr("height", d => height - y(d[2]))
            .attr("class", (d,i) => `attribution-score-histogram-bar attribution-score-histogram-bar-${i}`)
            .style("fill", fillColor)

    svg.append("g")
        .attr("class", "attribution-score-histogram-bar-hovered-wrapper")
        .selectAll("rect")
        .data(counts)
        .enter()
        .append("rect")
            .attr("x", 0)
            .attr("transform", d => {
                return `translate(${x(d[0])}, 0)`})
            .attr("width", d => x(d[1]) - x(d[0]) - 0)
            .attr("height", d => height)
            .attr("class", (d,i) => `attribution-score-histogram-bar-hovered attribution-score-histogram-bar-hovered-${i}`)
            .style("fill", "#ffffff00")
            .on("mouseover", function(event) {
                let flag = false
                let bar_index = +this.classList[1].split("-").pop()
                // loop through all the positive attributed text and highlight in blue as well as the corresponding text box if any
                for (let i=0; i<num_pos_shown; i++) {
                    let attributedTextWrapperElement = document.getElementById(`positive-attributed-text-wrapper-${i}`)
                    let data = attributedTextWrapperElement.attributionData
                    if (data["score_histogram_bin"] == bar_index) {
                        d3.select(`#positive-attributed-text-wrapper-${i}`).style("background-color", "#abd9e970")
                        histogramHighlightedTextWrapperElementIds.push(`positive-attributed-text-wrapper-${i}`)
                        flag = true
                    }
                }
                if (flag) {
                    // d3.select(this).style("fill", "#abd9e9")
                    d3.select(`.attribution-score-histogram-bar-${bar_index}`).style("fill", "#64c0f1e0")
                    return
                }
                
                // loop through all the negative attributed text and highlight in red as well as the corresponding the corresponding box if any
                for (let i=0; i<num_neg_shown; i++) {
                    let attributedTextWrapperElement = document.getElementById(`negative-attributed-text-wrapper-${i}`)
                    let data = attributedTextWrapperElement.attributionData
                    if (data["score_histogram_bin"] == bar_index) {
                        d3.select(`#negative-attributed-text-wrapper-${i}`).style("background-color", "#fdae6150")
                        histogramHighlightedTextWrapperElementIds.push(`negative-attributed-text-wrapper-${i}`)
                        flag = true
                    }
                }
                if (flag) {
                    // d3.select(this).style("fill", "#fdae61b0")
                    d3.select(`.attribution-score-histogram-bar-${bar_index}`).style("fill", "#fdae61b0")
                    return
                }
            })
            .on("mouseout", function(event) {
                // d3.select(this).style("fill", fillColor)
                let bar_index = +this.classList[1].split("-").pop()
                d3.select(`.attribution-score-histogram-bar-${bar_index}`).style("fill", fillColor)
                // revert the color of the corresponding text box
                for (let i=0; i<histogramHighlightedTextWrapperElementIds.length; i++) {
                    let id = histogramHighlightedTextWrapperElementIds[i]
                    d3.select(`#${id}`).style("background-color", id.split("-")[0]=="positive"?"#e0f3f88a":"#fdae6120")
                    d3.select(`#${histogramHighlightedTextWrapperElementIds[i]}`).style("background-color", "")
                }
                histogramHighlightedTextWrapperElementIds = []
            })


    // let attributedTextWrapperElement = document.getElementById(`${type}-attributed-text-wrapper-${i}`) --> it has corresponding bin number 

}

function resetAttributedDataSummary(posTfIdf, negTfIdf) {
    // TODO: visualize TF-IDF data (when num_shown for pos is 3, we can use the data at [3])
    // num_pos_shown, num_neg_shown
    summarizeAttributedData(posTfIdf, positive=true)
    summarizeAttributedData(negTfIdf, positive=false)
    adjustSidebarHeight()
}

function summarizeAttributedData(tfIdf, positive=true) {
    let type = positive?"positive":"negative";
    d3.select(`.${type}-attribution-text-tf-idf`).selectAll("div").remove()
    let numShown = positive?num_pos_shown:num_neg_shown
    if (numShown==0) {
        d3.select(`.${type}-attribution-text-tf-idf`)
            .append("div")
            .attr("class", `${type}-attribution-text-tf-idf-row attribution-text-tf-idf-row`)
            .text("No data to show")
        return
    }

    data = tfIdf[numShown-1]
    d3.select(`.${type}-attribution-text-tf-idf`)
        .selectAll("div")
        .data(data)
        .enter()
        .append("div")
            .attr("class", `${type}-attribution-text-tf-idf-word attribution-text-tf-idf-word`)
            // .html(d => `<div class="attribution-text-tf-idf-word">${d[0]}</div><div class="attribution-text-tf-idf-value">${d[1]}</div>`)
            .text(d => d[0])
}

function adjustSidebarHeight() {
    let sidebarHeight = document.getElementsByClassName("sidebar")[0].getBoundingClientRect().height
    let attributionHeight = document.getElementsByClassName("attribution-result-wrapper")[0].getBoundingClientRect().height
    console.log(sidebarHeight, attributionHeight)
    let height = Math.max(sidebarHeight, attributionHeight)
    document.getElementsByClassName("sidebar")[0].style.height = `${height}px`
}