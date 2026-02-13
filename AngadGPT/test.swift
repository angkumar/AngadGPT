import Foundation

let url = URL(string: "http://localhost:8000/generate")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")

let payload = ["prompt": "Once upon a time,", "api_key": "ILoveYaana"]

request.httpBody = try! JSONSerialization.data(withJSONObject: payload)

let task = URLSession.shared.dataTask(with: request) { data, response, error in
    guard let data = data else { return }
    let json = try! JSONSerialization.jsonObject(with: data, options: [])
    print(json)
}

task.resume()