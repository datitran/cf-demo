require "sinatra"

set :public_folder, "public"

@@hostname = ENV["PREDICTION_API"] 

get "/" do
  erb :index
end
