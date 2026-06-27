// Retail Analytics & ML Dashboard JS Engine (Lightweight & CORS-Free)

document.addEventListener('DOMContentLoaded', () => {
    // Global state
    let edaData = null;
    let mlData = null;

    // Load Data
    const loadDashboardData = () => {
        try {
            // Read from global window variables loaded via data.js (CORS-free)
            edaData = window.edaData;
            mlData = window.mlData;

            if (!edaData || !mlData) {
                throw new Error("Retail data variables (window.edaData, window.mlData) are missing. Make sure data.js is generated and loaded correctly.");
            }

            initOverview();
            initEDATab();
            initSegmentsTab();
            initPredictTab();
            
            console.log("Dashboard data loaded successfully from data.js.");
        } catch (error) {
            console.error("Error loading dashboard data:", error);
            alert("Could not load dataset variables. Please make sure you run data_analysis.py and machine_learning.py to generate assets/data/data.js first!");
        }
    };

    // Tab Navigation Logic
    const menuItems = document.querySelectorAll('.menu-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    const tabDetails = {
        'overview': {
            title: 'Dashboard Overview',
            subtitle: 'Overview of key metrics and sales performance trends.'
        },
        'eda': {
            title: 'Exploratory Data Analysis Journey',
            subtitle: 'A detailed statistical study of consumer behaviors answering your 7 questions.'
        },
        'segments': {
            title: 'Customer Segments (ML)',
            subtitle: 'Unsupervised customer profiling based on age, spending, and purchase volume.'
        },
        'predict': {
            title: 'Predictive Simulator (ML)',
            subtitle: 'Instant transaction quantity and spending estimator powered by Ridge Regression.'
        }
    };

    menuItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            
            // Toggle active buttons
            menuItems.forEach(btn => btn.classList.remove('active'));
            item.classList.add('active');

            // Toggle active sections
            tabContents.forEach(tab => tab.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');

            // Update Titles
            pageTitle.textContent = tabDetails[targetTab].title;
            pageSubtitle.textContent = tabDetails[targetTab].subtitle;
        });
    });

    // --- Tab 1: Overview Dashboard Initialization ---
    const initOverview = () => {
        if (!edaData) return;

        const stats = edaData.general_stats;
        
        // Update header range
        document.getElementById('dataset-range').textContent = `${stats.date_range[0]} to ${stats.date_range[1]}`;

        // Populate KPIs
        document.getElementById('stat-revenue').textContent = `$${stats.total_revenue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        document.getElementById('stat-transactions').textContent = stats.total_transactions.toLocaleString();
        document.getElementById('stat-avg-spend').textContent = `$${stats.average_spend_per_transaction.toFixed(2)}`;
        document.getElementById('stat-avg-age').textContent = stats.avg_age.toFixed(1);
    };

    // --- Tab 2: EDA Journey Initialization ---
    const initEDATab = () => {
        if (!edaData) return;

        const select = document.getElementById('question-select');
        const questions = document.querySelectorAll('.eda-question-content');

        const populateQ7Table = () => {
            const tbody = document.querySelector('#table-q7 tbody');
            if (tbody && tbody.children.length === 0) {
                const stats = edaData.q7_price_stats;
                stats.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td><strong>${row['Product Category']}</strong></td>
                        <td>${row.count}</td>
                        <td>$${row.min.toFixed(2)}</td>
                        <td>$${row['50%'].toFixed(2)}</td>
                        <td>$${row.max.toFixed(2)}</td>
                        <td>$${row.std.toFixed(2)}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        };

        select.addEventListener('change', () => {
            const qId = select.value;
            questions.forEach(q => q.classList.remove('active'));
            document.getElementById(`eda-${qId}`).classList.add('active');
            
            if (qId === 'q7') {
                populateQ7Table();
            }
        });

        // Run initially for selected question
        if (select.value === 'q7') {
            populateQ7Table();
        }
    };

    // --- Tab 3: Customer Segments (K-Means Clustering) ---
    const initSegmentsTab = () => {
        if (!mlData) return;

        const profiles = mlData.clustering.cluster_profiles;
        const container = document.getElementById('segments-container');
        container.innerHTML = ''; // Clear previous

        profiles.forEach(prof => {
            const card = document.createElement('div');
            card.className = 'segment-card';
            card.innerHTML = `
                <h4>${prof.name}</h4>
                <p class="segment-desc">${prof.description}</p>
                <div class="segment-stats">
                    <div class="segment-stat">
                        <span>Avg Age</span>
                        <strong>${prof.avg_age.toFixed(1)} y/o</strong>
                    </div>
                    <div class="segment-stat">
                        <span>Avg Spend</span>
                        <strong>$${prof.avg_spend.toFixed(2)}</strong>
                    </div>
                    <div class="segment-stat">
                        <span>Avg Qty</span>
                        <strong>${prof.avg_quantity.toFixed(2)}</strong>
                    </div>
                    <div class="segment-stat">
                        <span>Segment %</span>
                        <strong>${prof.percentage.toFixed(1)}%</strong>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });

        // Form elements - Segment Checker
        const form = document.getElementById('segment-form');
        const resultBox = document.getElementById('segment-result');
        const resultName = document.getElementById('result-segment-name');
        const resultDesc = document.getElementById('result-segment-desc');

        const ageInput = document.getElementById('seg-age');
        const spendInput = document.getElementById('seg-spend');
        const qtyInput = document.getElementById('seg-qty');

        // Function to run matching calculation
        const runSegmentMatching = () => {
            const age = parseFloat(ageInput.value);
            const spend = parseFloat(spendInput.value);
            const qty = parseFloat(qtyInput.value);

            if (isNaN(age) || isNaN(spend) || isNaN(qty)) {
                return; // Wait until all are filled
            }

            // Scale inputs based on training metrics
            const mean = mlData.clustering.scaler_mean;
            const std = mlData.clustering.scaler_scale;

            const scaledAge = (age - mean[0]) / std[0];
            const scaledSpend = (spend - mean[1]) / std[1];
            const scaledQty = (qty - mean[2]) / std[2];

            const scaledInput = [scaledAge, scaledSpend, scaledQty];

            // Calculate Euclidean distance to each cluster centroid
            const centroids = mlData.clustering.centroids;
            let closestCluster = 0;
            let minDistance = Infinity;

            centroids.forEach((centroid, idx) => {
                let sumSq = 0;
                for (let j = 0; j < 3; j++) {
                    sumSq += Math.pow(scaledInput[j] - centroid[j], 2);
                }
                const dist = Math.sqrt(sumSq);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestCluster = idx;
                }
            });

            // Retrieve segment details
            const matchedProfile = profiles.find(p => p.cluster_id === closestCluster);
            if (!matchedProfile) return;

            // Output UI
            resultName.textContent = matchedProfile.name;
            resultDesc.textContent = matchedProfile.description;
            
            // Adjust card accent colors dynamically in output (robust to any cluster count)
            const accentColors = ['#2563eb', '#10b981', '#f59e0b', '#ec4899', '#a855f7', '#fb7185', '#06b6d4'];
            const color = accentColors[closestCluster % accentColors.length];
            resultName.style.color = color;
            resultBox.style.borderColor = color + '44';
            resultBox.style.backgroundColor = color + '11';

            resultBox.classList.remove('hidden');
        };

        // Form Submission
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            runSegmentMatching();
        });

        // Add event listeners for instant reactive updates when inputs are changed
        [ageInput, spendInput, qtyInput].forEach(input => {
            input.addEventListener('input', runSegmentMatching);
            input.addEventListener('change', runSegmentMatching);
        });

        // Run initially
        runSegmentMatching();
    };

    // --- Tab 4: Predictive Simulator (Ridge Regression) ---
    const initPredictTab = () => {
        if (!mlData) return;

        const reg = mlData.regression;

        // Render metrics
        document.getElementById('model-qty-r2').textContent = reg.quantity_model.r2.toFixed(3);
        document.getElementById('model-qty-mae').textContent = `${reg.quantity_model.mae.toFixed(2)} items`;
        document.getElementById('model-amt-r2').textContent = reg.total_amount_model.r2.toFixed(3);
        document.getElementById('model-amt-mae').textContent = `$${reg.total_amount_model.mae.toFixed(2)}`;

        // Adjust class if positive or negative
        const qtyR2El = document.getElementById('model-qty-r2');
        const amtR2El = document.getElementById('model-amt-r2');
        if (reg.quantity_model.r2 >= 0) qtyR2El.className = "metric-val text-purple";
        if (reg.total_amount_model.r2 >= 0) amtR2El.className = "metric-val text-purple";

        // Render active weights list (e.g. for Total Spend Model)
        const weightsList = document.getElementById('coefficients-list');
        weightsList.innerHTML = '';

        const coefs = reg.total_amount_model.coefficients;
        Object.entries(coefs).forEach(([feature, val]) => {
            const row = document.createElement('div');
            row.className = 'weight-row';
            const cleanFeatureName = feature.replace('_', ': ');
            
            let valClass = 'positive';
            let valSymbol = '+';
            if (val < 0) {
                valClass = 'negative';
                valSymbol = '';
            }

            row.innerHTML = `
                <span class="weight-name">${cleanFeatureName}</span>
                <span class="weight-value ${valClass}">${valSymbol}${val.toFixed(3)}</span>
            `;
            weightsList.appendChild(row);
        });

        // Form elements
        const form = document.getElementById('predict-form');
        const qtyOutput = document.getElementById('pred-output-qty');
        const amtOutput = document.getElementById('pred-output-amt');

        const dateInput = document.getElementById('pred-date');
        const ageInput = document.getElementById('pred-age');
        const genderSelect = document.getElementById('pred-gender');
        const categorySelect = document.getElementById('pred-category');

        const runPrediction = () => {
            const dateVal = new Date(dateInput.value);
            const age = parseFloat(ageInput.value);
            const gender = genderSelect.value;
            const category = categorySelect.value;

            if (isNaN(dateVal.getTime()) || isNaN(age)) {
                return;
            }

            // Parse Date components
            const month = dateVal.getMonth() + 1; // 1-12
            const year = dateVal.getFullYear();
            
            const daysOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
            const dayOfWeekName = daysOfWeek[dateVal.getDay()];

            // Construct feature vector based on ml_results feature order
            const features = reg.feature_names;
            const inputVector = {};

            // Initialize all features as 0
            features.forEach(f => {
                inputVector[f] = 0;
            });

            // Set numeric features
            inputVector['Age'] = age;
            inputVector['Month'] = month;
            inputVector['Year'] = year;

            // Set dummy columns matching selections
            const genderCol = `Gender_${gender}`;
            const categoryCol = `Product_Category_${category}`;
            const dayCol = `DayOfWeek_${dayOfWeekName}`;

            if (genderCol in inputVector) inputVector[genderCol] = 1;
            if (categoryCol in inputVector) inputVector[categoryCol] = 1;
            if (dayCol in inputVector) inputVector[dayCol] = 1;

            // Compute Quantity prediction (Intercept + dot product)
            const qtyModel = reg.quantity_model;
            let predQty = qtyModel.intercept;
            
            features.forEach(f => {
                predQty += inputVector[f] * (qtyModel.coefficients[f] || 0);
            });

            // Compute Amount prediction
            const amtModel = reg.total_amount_model;
            let predAmt = amtModel.intercept;
            
            features.forEach(f => {
                predAmt += inputVector[f] * (amtModel.coefficients[f] || 0);
            });

            // Bound values to realistic thresholds
            predQty = Math.max(1.0, predQty);
            predAmt = Math.max(10.0, predAmt);

            // Update UI elements smoothly
            qtyOutput.textContent = predQty.toFixed(2) + ' items';
            amtOutput.textContent = '$' + predAmt.toFixed(2);
        };

        // Form Submit
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            runPrediction();
        });

        // Add event listeners for instant reactive recalculations
        [dateInput, ageInput, genderSelect, categorySelect].forEach(el => {
            el.addEventListener('input', runPrediction);
            el.addEventListener('change', runPrediction);
        });

        // Run initially
        runPrediction();
    };

    // Load initial data
    loadDashboardData();
});
