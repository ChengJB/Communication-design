clear all;close all;clc;
% 比特流生成
number=10^4; 
bit_stream=[randi([0 1],number,1);[0;0]];
codeword=zeros(2*(number+2),1); %卷积码序列
D1=0; % 寄存器初始状态
D2=0;
for i_b=1:1:number+2
y1=mod(bit_stream(i_b)+D1+D2,2);
y2=mod(bit_stream(i_b)+D2,2); %计算每个输入对应的两个输出
D2=D1; 
D1=bit_stream(i_b); %开始移位
codeword(2*i_b-1)=y1;
codeword(2*i_b)=y2; %以y1y2的顺序存储进Codeword
end
codeword;
%ISI信道
data_len = 2*(number+2); % 数据流长度
SNR_dB = 1; % 单位比特信噪比
chan_len = 3; % 通道抽头数
fade_var = 1; % 信道的衰减方差
decoding_delay = 10; % 维特比算法的译码延迟
SNR = 10^(0.1*SNR_dB); 
noise_var = 1*(fade_var*chan_len)/(2*SNR); % 信噪比参数 
chan_input = codeword'; % 信道输入,将卷积码作为信道输入 
bpsk_seq = 1-2*chan_input; % bpsk
fade_chan = normrnd(0,sqrt(fade_var),1,chan_len);% ISI信道的频率响应脉冲
noise = normrnd(0,sqrt(noise_var),1,data_len+chan_len-1);% awgn
chan_op = conv(fade_chan,bpsk_seq)+noise; % 信道输出
%viterbi译码
steady_state = chan_op(chan_len:data_len);
% branch metrics for the Viterbi algorithm
branch_metric = zeros(2^chan_len,data_len-chan_len+1);
branch_metric(1,:) = (steady_state-(fade_chan(1)+fade_chan(2)+fade_chan(3))).^2;
branch_metric(2,:) = (steady_state-(fade_chan(1)+fade_chan(2)-fade_chan(3))).^2;
branch_metric(3,:) = (steady_state-(fade_chan(1)-fade_chan(2)+fade_chan(3))).^2;
branch_metric(4,:) = (steady_state-(fade_chan(1)-fade_chan(2)-fade_chan(3))).^2;
branch_metric(5,:) = (steady_state-(-fade_chan(1)+fade_chan(2)+fade_chan(3))).^2;
branch_metric(6,:) = (steady_state-(-fade_chan(1)+fade_chan(2)-fade_chan(3))).^2;
branch_metric(7,:) = (steady_state-(-fade_chan(1)-fade_chan(2)+fade_chan(3))).^2;
branch_metric(8,:) = (steady_state-(-fade_chan(1)-fade_chan(2)-fade_chan(3))).^2;
%-----------------------------------------------------------------------------------
dec_a=Viterbi_algorithm(data_len-chan_len+1,decoding_delay,branch_metric);
% 误比特率
ber = nnz(chan_input(chan_len:data_len-decoding_delay)-dec_a )/(data_len-chan_len+1-decoding_delay)
