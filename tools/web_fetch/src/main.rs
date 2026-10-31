use std::{collections::HashSet, io::{self, Read, Write}, net::{IpAddr, SocketAddr}, time::Duration};
use serde::{Deserialize, Serialize};
use url::{Url, Host};
use reqwest::{Client, redirect::Policy, header};
use scraper::{Html, Selector};

const BODY_MAX: usize = 2_097_152;
const TEXT_MAX: usize = 65_536;
const STDOUT_MAX: usize = 131_072;
#[derive(Debug)]
struct Failure { code: &'static str, message: String }
fn err(code: &'static str, message: impl Into<String>) -> Failure { Failure {code, message:message.into()} }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request { version: u32, action: String, url: String }
#[derive(Serialize)]
struct Page { requested_url: String, final_url: String, status: u16, content_type: String,
    title: Option<String>, text: String, truncated: bool }
#[derive(Serialize)]
struct ErrorBody { code: &'static str, message: String }
#[derive(Serialize)]
struct Response { version: u8, ok: bool, data: Option<Page>, error: Option<ErrorBody> }
impl Response {
    fn failed(e: Failure) -> Self { Self {version:1,ok:false,data:None,error:Some(ErrorBody{code:e.code,message:e.message})} }
    fn success(p: Page) -> Self { Self {version:1,ok:true,data:Some(p),error:None} }
}
fn trim_bytes(s: &mut String, max: usize) -> bool {
    if s.len() <= max { return false; }
    let mut n = max;
    while !s.is_char_boundary(n) { n -= 1; }
    s.truncate(n); true
}
fn normalize_host(s: &str) -> Result<String, Failure> {
    if s.trim() != s || s.is_empty() || s.ends_with('.') {
        return Err(err("URL_BLOCKED", "Invalid or trailing-dot hostname"));
    }
    match Host::parse(s).map_err(|_| err("URL_BLOCKED", "Invalid hostname"))? {
        Host::Domain(d) => Ok(d.to_ascii_lowercase()),
        _ => Err(err("URL_BLOCKED", "IP-literal hosts are not allowed")),
    }
}
fn config() -> Result<HashSet<String>, Failure> {
    if std::env::var("WEB_FETCH_ENABLED").as_deref() != Ok("1") {
        return Err(err("PERMISSION_DENIED", "Web fetching is disabled"));
    }
    let raw = std::env::var("WEB_FETCH_ALLOWED_HOSTS_JSON")
        .map_err(|_| err("PERMISSION_DENIED", "Missing allowlist"))?;
    if raw.len() > 16384 { return Err(err("PERMISSION_DENIED", "Allowlist too large")); }
    let list: Vec<String> = serde_json::from_str(&raw)
        .map_err(|_| err("PERMISSION_DENIED", "Invalid allowlist JSON"))?;
    if list.is_empty() || list.len() > 100 { return Err(err("PERMISSION_DENIED", "Empty or oversized allowlist")); }
    list.iter().map(|s| normalize_host(s).map_err(|_| err("PERMISSION_DENIED", "Invalid allowlist host"))).collect()
}
fn validate_url(raw: &str, allowed: &HashSet<String>) -> Result<Url, Failure> {
    if raw.is_empty() || raw.len()>4096 || raw.chars().any(|c| c.is_whitespace() || c.is_control()) || raw.contains('\\') {
        return Err(err("INVALID_REQUEST", "Invalid URL length or characters"));
    }
    let u = Url::parse(raw).map_err(|_| err("INVALID_REQUEST", "Invalid absolute URL"))?;
    let port = match u.scheme() { "http" => 80, "https" => 443,
        _ => return Err(err("URL_BLOCKED", "Only HTTP and HTTPS are allowed")) };
    if !u.username().is_empty() || u.password().is_some() || u.port_or_known_default()!=Some(port) {
        return Err(err("URL_BLOCKED", "Credentials or nonstandard port"));
    }
    let h=normalize_host(u.host_str().ok_or_else(||err("URL_BLOCKED","Missing host"))?)?;
    if !allowed.contains(&h) { return Err(err("URL_BLOCKED", "Hostname is not allowlisted")); }
    Ok(u)
}
fn public_ip(ip: IpAddr) -> bool {
    // Conservative exclusions: not a complete assertion that an address is safe.
    // Only ordinary globally-routable unicast is supported.
    let blocked = match ip {
        IpAddr::V4(_) => vec!["0.0.0.0/8","10.0.0.0/8","100.64.0.0/10","127.0.0.0/8",
            "169.254.0.0/16","172.16.0.0/12","192.0.0.0/24","192.0.2.0/24",
            "192.88.99.0/24","192.168.0.0/16","198.18.0.0/15","198.51.100.0/24",
            "203.0.113.0/24","224.0.0.0/3"],
        IpAddr::V6(v) => {
            let global: ipnet::IpNet = "2000::/3".parse().unwrap();
            if !global.contains(&IpAddr::V6(v)) { return false; }
            vec!["2001::/23","2001:db8::/32","2002::/16","3fff::/20"]
        }
    };
    !blocked.iter().any(|s| s.parse::<ipnet::IpNet>().unwrap().contains(&ip))
}
async fn addresses(u: &Url) -> Result<Vec<SocketAddr>, Failure> {
    let addrs: Vec<_> = tokio::net::lookup_host((u.host_str().unwrap(), u.port_or_known_default().unwrap()))
        .await.map_err(|_|err("DNS_ERROR","DNS resolution failed"))?.collect();
    if addrs.is_empty() { return Err(err("DNS_ERROR","No destination addresses")); }
    if addrs.iter().any(|a|!public_ip(a.ip())) { return Err(err("URL_BLOCKED","DNS returned a non-public address")); }
    Ok(addrs)
}
fn extract(body: &[u8], kind: &str) -> (Option<String>,String,bool) {
    let decoded=String::from_utf8_lossy(body);
    let (mut title,mut text)=if kind=="text/html" {
        let doc=Html::parse_document(&decoded);
        let title_sel=Selector::parse("title").unwrap();
        let title=doc.select(&title_sel).next().map(|e|e.text().collect::<Vec<_>>().join(" "));
        let mut pieces=Vec::new();
        for node in doc.tree.nodes() {
            if let Some(t)=node.value().as_text() {
                let excluded=node.ancestors().any(|a| a.value().as_element()
                    .map(|e| matches!(e.name(),"script"|"style"|"head"|"noscript"|"template"))
                    .unwrap_or(false));
                if !excluded { pieces.push(t.to_string()); }
            }
        }
        (title,pieces.join(" ").split_whitespace().collect::<Vec<_>>().join(" "))
    } else { (None,decoded.into_owned()) };
    let mut truncated=trim_bytes(&mut text,TEXT_MAX);
    if let Some(t)=title.as_mut() { truncated |= trim_bytes(t,1024); }
    (title,text,truncated)
}
async fn fetch(request: Request, allowed: &HashSet<String>) -> Result<Page, Failure> {
    let mut u=validate_url(&request.url,allowed)?;
    for hop in 0..=3 {
        let addrs=addresses(&u).await?;
        let client=Client::builder().no_proxy().redirect(Policy::none())
            .resolve_to_addrs(u.host_str().unwrap(),&addrs)
            .timeout(Duration::from_secs(15)).build()
            .map_err(|_|err("INTERNAL_ERROR","Cannot build HTTP client"))?;
        let mut response=client.get(u.clone()).header(header::ACCEPT_ENCODING,"identity")
            .header(header::ACCEPT,"text/html, text/plain")
            .header(header::USER_AGENT,"web_fetch/0.1")
            .send().await.map_err(|e|err(if e.is_timeout(){"TIMEOUT"}else{"NETWORK_ERROR"},"HTTP request failed"))?;
        let status=response.status();
        if matches!(status.as_u16(),301|302|303|307|308) {
            if hop==3 { return Err(err("TOO_MANY_REDIRECTS","Redirect limit reached")); }
            let target=response.headers().get(header::LOCATION).and_then(|h|h.to_str().ok())
                .ok_or_else(||err("HTTP_ERROR","Missing or invalid redirect location"))?;
            let next=u.join(target).map_err(|_|err("URL_BLOCKED","Invalid redirect URL"))?;
            u=validate_url(next.as_str(),allowed)?;
            continue;
        }
        if !status.is_success() { return Err(err("HTTP_ERROR",format!("HTTP status {}",status.as_u16()))); }
        if response.headers().get_all(header::CONTENT_ENCODING).iter().any(|v|v.to_str().map(|s|!s.eq_ignore_ascii_case("identity")).unwrap_or(true)) {
            return Err(err("UNSUPPORTED_CONTENT_TYPE","Encoded response bodies are not supported"));
        }
        let content_type=response.headers().get(header::CONTENT_TYPE).and_then(|v|v.to_str().ok())
            .unwrap_or("").split(';').next().unwrap_or("").trim().to_ascii_lowercase();
        if content_type!="text/html" && content_type!="text/plain" {
            return Err(err("UNSUPPORTED_CONTENT_TYPE","Only HTML and plain text are supported"));
        }
        if response.content_length().map(|n|n>BODY_MAX as u64).unwrap_or(false) {
            return Err(err("RESPONSE_TOO_LARGE","Response body exceeds limit"));
        }
        let mut body=Vec::new();
        while let Some(chunk)=response.chunk().await.map_err(|e|err(if e.is_timeout(){"TIMEOUT"}else{"NETWORK_ERROR"},"Body read failed"))? {
            if body.len()+chunk.len()>BODY_MAX { return Err(err("RESPONSE_TOO_LARGE","Response body exceeds limit")); }
            body.extend_from_slice(&chunk);
        }
        let (title,text,truncated)=extract(&body,&content_type);
        return Ok(Page {requested_url:request.url,final_url:u.to_string(),status:status.as_u16(),content_type,title,text,truncated});
    }
    Err(err("TOO_MANY_REDIRECTS","Redirect limit reached"))
}
fn encode(mut response: Response) -> Vec<u8> {
    loop {
        let mut bytes=serde_json::to_vec(&response).expect("Serializable response");
        bytes.push(b'\n');
        if bytes.len()<=STDOUT_MAX { return bytes; }
        if let Some(page)=response.data.as_mut() {
            if !page.text.is_empty() { let n=page.text.len()/2;trim_bytes(&mut page.text,n);page.truncated=true;continue; }
        }
        response=Response::failed(err("RESPONSE_TOO_LARGE","JSON response exceeds limit"));
    }
}
#[tokio::main]
async fn main() {
    let result=async {
        let allowed=config()?;
        let mut input=Vec::new();
        io::stdin().take(8193).read_to_end(&mut input).map_err(|_|err("INVALID_REQUEST","Cannot read stdin"))?;
        if input.len()>8192 { return Err(err("INVALID_REQUEST","Input exceeds limit")); }
        let request:Request=serde_json::from_slice(&input).map_err(|_|err("INVALID_REQUEST","Invalid request JSON"))?;
        if request.version!=1 || request.action!="web_fetch" { return Err(err("INVALID_REQUEST","Unsupported version or action")); }
        tokio::time::timeout(Duration::from_secs(15),fetch(request,&allowed)).await
            .map_err(|_|err("TIMEOUT","Total fetch deadline exceeded"))?
    }.await;
    let response=match result {Ok(p)=>Response::success(p),Err(e)=>Response::failed(e)};
    let success=response.ok;
    if io::stdout().write_all(&encode(response)).is_err() { std::process::exit(1); }
    std::process::exit(if success {0}else{1});
}
#[cfg(test)]
mod tests {
    use super::*;
    fn allowed()->HashSet<String>{["example.com".to_string()].into_iter().collect()}
    #[test] fn validates_url() {
        assert!(validate_url("https://example.com/",&allowed()).is_ok());
        for url in ["http://127.0.0.1/","http://localhost/","file:///etc/passwd","https://example.com:8080/","https://user:pass@example.com/","https://evil.example/","https://example.com./"] {
            assert!(validate_url(url,&allowed()).is_err(),"{url}");
        }
    }
    #[test] fn rejects_private_and_special_ips() {
        for ip in ["127.0.0.1","10.0.0.1","169.254.169.254","100.64.0.1","192.0.2.1","224.0.0.1","::1","::ffff:8.8.8.8","fc00::1","fe80::1","2001:db8::1"] { assert!(!public_ip(ip.parse().unwrap()),"{ip}"); }
        assert!(public_ip("8.8.8.8".parse().unwrap()));
        assert!(public_ip("2606:4700:4700::1111".parse().unwrap()));
    }
    #[test] fn removes_scripts_and_styles() {
        let (title,text,_)=extract(b"<html><head><title>Title</title><style>bad css</style></head><body>Hello<script>bad js</script><p>world</p></body></html>","text/html");
        assert_eq!(title.as_deref(),Some("Title"));assert!(text.contains("Hello"));assert!(!text.contains("bad"));
    }
    #[test] fn utf8_and_json_limits() {
        let mut t="\u{1f600}".repeat(40000); assert!(trim_bytes(&mut t,65535));assert!(t.len()<=65535);
        let p=Page{requested_url:"https://example.com/".into(),final_url:"https://example.com/".into(),status:200,content_type:"text/plain".into(),title:None,text:"\u{0001}".repeat(TEXT_MAX),truncated:false};
        let bytes=encode(Response::success(p));assert!(bytes.len()<=STDOUT_MAX);let v:serde_json::Value=serde_json::from_slice(&bytes).unwrap();assert_eq!(v["data"]["truncated"],true);
    }
    #[test] fn rejects_extra_fields_and_boolean_version() {
        assert!(serde_json::from_str::<Request>(r#"{"version":true,"action":"web_fetch","url":"https://example.com"}"#).is_err());
        assert!(serde_json::from_str::<Request>(r#"{"version":1,"action":"web_fetch","url":"https://example.com","exec":"sh"}"#).is_err());
    }
}
