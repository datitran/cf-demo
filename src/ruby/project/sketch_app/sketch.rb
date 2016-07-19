require "sinatra"

set :public_folder, "public"

if ENV["PREDICTION_API"] != nil
  @@hostname = ENV["PREDICTION_API"]
else
  @@hostname = "http://number-predictor-testing.local.pcfdev.io"
end

get "/" do
  erb :index
end
