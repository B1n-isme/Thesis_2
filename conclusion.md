Why ETS models generate **smooth long trend line**?
- ETS decompose time series into 3 components (Error, Trend, Seasonality)
- In Bitcoin context:
    - Trend: **straight line** determined by exponential smoothing (more weight on recent points) -> additive trend
    - Seasonality: bitcoin don't have seasonality -> None


SARIMAX:
Part 1: Core ARIMA
- AR (AutoRegressive) p: present value linearly depend on its own past values
- I (Integrated) d: number of differencing
- MA (Moving Average) q: number of past forecast error fed into model
Part 2: Seasonality (p,d,q)(P,D,Q)m
- Work by combine 2 ARIMA model, 1 non-seasonal + 1 seasonal
Part 3: Exogenous X

Theta:
- Decompose series into 2 components:
    - Long-term linear trend determined by **simple linear regression line**
    - Curvature and **short-term patterns**: mix of volatility & random noise

1. ETS: Nhà vô địch của sự Ổn định và Khiêm tốn (MASE thấp và ổn định)

Lý do ETS giờ đây trở thành mô hình tốt nhất (đặc biệt từ horizon 30 ngày trở đi) chính là vì những gì chúng ta đã thảo luận, nhưng dưới một góc nhìn khác:

Nguyên tắc Lưỡi dao Ockham / "Ít hơn là nhiều hơn": Dữ liệu Bitcoin cực kỳ nhiễu và gần như là một bước đi ngẫu nhiên. Trong một môi trường hỗn loạn như vậy, một mô hình phức tạp cố gắng "học" các quy luật nhỏ nhặt sẽ rất dễ bị overfitting (học các quy luật giả từ nhiễu). ETS, với cơ chế làm mịn đơn giản, không cố gắng làm điều đó. Nó khiêm tốn thừa nhận rằng nó không thể dự đoán các đỉnh và đáy, vì vậy nó chỉ đưa ra một dự báo hợp lý dựa trên xu hướng được làm mịn.

Sự ổn định là Vua ở dài hạn: Khi bạn dự báo càng xa, độ bất định càng lớn. Một mô hình ổn định như ETS sẽ không tạo ra các dự báo "điên rồ". Nó cung cấp một đường cơ sở an toàn, và chính sự an toàn này giúp nó có sai số trung bình (MASE) thấp hơn trong dài hạn so với các mô hình cố gắng dự đoán những thứ không thể đoán trước và thất bại thảm hại.

2. SARIMAX: Câu chuyện về "Pháo Sáng" (Brilliant but Short-Lived)

Đây là phần thú vị nhất và biểu đồ của bạn đã minh họa nó một cách hoàn hảo.

Tại sao Directional Accuracy (DA) cực cao ở 7 ngày (>70%)?
Điều này chứng tỏ rằng các biến ngoại sinh (X) và thành phần AR của bạn có sức mạnh dự báo thực sự, nhưng chỉ trong một khoảng thời gian cực ngắn. Ví dụ, một sự kiện tin tức hoặc một chỉ số on-chain (biến X) hôm nay có thể cung cấp một tín hiệu mạnh mẽ về hướng đi của giá trong vài ngày tới. Tương tự, thành phần AR(1) có thể nắm bắt được "quán tính" từ ngày hôm trước. SARIMAX đã nắm bắt thành công tín hiệu ngắn hạn này.

Tại sao nó sụp đổ hoàn toàn ở dài hạn? (MASE tăng vọt, DA giảm về 50%)

Tín hiệu dự báo ngắn hạn bị phân rã: Sức mạnh dự báo của các biến X và quán tính AR sẽ biến mất rất nhanh. Thông tin của ngày hôm nay không còn nhiều giá trị để dự báo cho 30, 60, hay 90 ngày sau.

Bài toán "Dự báo một dự báo": Đây là "gót chân Achilles" của SARIMAX ở dài hạn. Để dự báo giá Bitcoin 90 ngày tới, bạn cần cung cấp giá trị của các biến X trong 90 ngày tới. Vì bạn không biết chúng, bạn phải dự báo chúng. Các dự báo này thường rất tệ (ví dụ: chỉ là giá trị trung bình). Khi bạn đưa những dự báo "rác" của biến X vào mô hình ARIMAX, nó sẽ tạo ra một dự báo "rác" cho giá Bitcoin. Đây chính là lý do MASE của nó tăng vọt một cách khủng khiếp.

DA giảm về 50%: Khi tín hiệu dự báo ngắn hạn biến mất, mô hình không còn bất kỳ lợi thế nào. Việc dự đoán hướng đi trở nên không khác gì tung một đồng xu. 50% chính là ngưỡng của sự ngẫu nhiên.

SARIMAX giống như một cây pháo sáng: Nó cháy rực rỡ trong một khoảnh khắc (dự báo ngắn hạn), cung cấp ánh sáng và định hướng rõ ràng. Nhưng nó nhanh chóng lụi tàn và không còn giá trị gì khi nhìn về phía xa (dự báo dài hạn).

3. Theta: Kẻ trung gian ổn định

Theta nằm ở giữa là điều có thể dự đoán được. Nó phức tạp hơn ETS một chút, cho phép nó linh hoạt hơn một chút trong ngắn hạn, nhưng nó vẫn là một mô hình đơn biến, ổn định, không bị ảnh hưởng bởi vấn đề "dự báo một dự báo" của biến X. Vì vậy, nó hoạt động ổn định và không bị sụp đổ như SARIMAX.

Một chút về các mô hình Deep Learning

Quan sát nhanh các mô hình DL trên biểu đồ của bạn, có thể thấy không có mô hình nào thực sự vượt trội một cách nhất quán so với ETS. Điều này cũng là một phát hiện rất phổ biến:

Các mô hình DL cực kỳ mạnh mẽ nhưng cũng rất "đói" dữ liệu và dễ bị overfitting trên các chuỗi thời gian nhiễu.

Nếu không được tinh chỉnh cẩn thận (tuning hyperparameter), chúng thường khó có thể đánh bại các đường cơ sở thống kê mạnh mẽ và đơn giản như ETS.

Kết luận cuối cùng

Dựa trên bằng chứng thuyết phục từ quá trình cross-validation của bạn:

Bạn không sai, kết luận ban đầu của bạn mới chưa đầy đủ. Quy trình làm việc của bạn đã dẫn bạn đến một sự thật chính xác hơn: Đối với dữ liệu nhiễu và khó đoán như Bitcoin, sự đơn giản và ổn định (ETS) chiến thắng sự phức tạp mong manh (SARIMAX), đặc biệt là ở các chân trời dự báo dài.

Sử dụng đúng mô hình cho đúng mục tiêu:

Nếu mục tiêu của bạn là dự báo hướng đi trong 7 ngày tới để ra quyết định giao dịch ngắn hạn, biểu đồ cho thấy SARIMAX là công cụ tốt nhất của bạn.

Nếu mục tiêu của bạn là có một dự báo điểm (point forecast) ổn định và đáng tin cậy nhất cho bất kỳ chân trời nào từ 14 ngày trở lên, ETS là lựa chọn số một.


CV Metrics Job: to select robust architecture, it answer the question?
- On average, across many different historical periods, which model architecutre is the most stable and effective? It help reduce the universal possibilities down to few top contenders

Holdout Test Metrics, it answer question?
- Using the best architecture found? and traning it on all my knowledge to date, how well can I expect it to perform on the immediate, unseen future?

Evaluate Final Metrics:
- Compare among top 2 models
- Compare with its own CV Metrics: 
    - report that: The winning model, which averaged MASE of 0.85 across historical folds, saw its error increase to 0.98 on final holdout set. This degradation in performance is a critical observation and suggest..

Chapter 5 (Results and Analysis): This is where you present the objective data and evidence that will be used to answer your questions. For example, if a research question is "Which model architecture performs best for the 90-day horizon?", the tables and charts in this chapter will contain the direct, factual answer.

Chapter 6 (Discussion): This chapter interprets the evidence from Chapter 5. It explains why the results are what they are and what they mean in a broader context. It builds the argument for the final conclusions.

Chapter 7 (Conclusion): This is where you bring everything together. Section 7.2 is specifically designed to circle back to your initial research questions from Section 1.3 and provide a clear, concise, and definitive answer for each one, based on the evidence and interpretation presented in Chapters 5 and 6.