# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
import schedule
import logging
from retry import retry
from github import Github
from github import Auth
import re
from datetime import datetime as dt
from os import getenv
from urllib.parse import urlparse, parse_qs

import requests
from certifi import where


TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
TOKEN = "github_pat_"
ACCOUNT_CSV_FILE = "data/accounts.csv"


class Auth0:
    def __init__(
        self,
        email: str,
        password: str,
        proxy: str = None,
        use_cache: bool = True,
        mfa: str = None,
    ):
        self.session_token = None
        self.email = email
        self.password = password
        self.use_cache = use_cache
        self.mfa = mfa
        self.session = requests.Session()
        self.req_kwargs = {
            "proxies": {
                "http": proxy,
                "https": proxy,
            }
            if proxy
            else None,
            "verify": where(),
            "timeout": 100,
        }
        self.access_token = None
        self.expires = None
        self.user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/109.0.0.0 Safari/537.36"
        )

    @staticmethod
    def __check_email(email: str):
        regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
        return re.fullmatch(regex, email)

    @retry(
        tries=3,
        delay=10,
    )
    def auth(self) -> str:
        if (
            self.use_cache
            and self.access_token
            and self.expires
            and self.expires > dt.now()
        ):
            return self.access_token

        if not self.__check_email(self.email) or not self.password:
            raise Exception("invalid email or password.")

        return self.__part_two()

    def __part_two(self) -> str:
        code_challenge = "w6n3Ix420Xhhu-Q5-mOOEyuPZmAsJHUbBpO8Ub7xBCY"
        code_verifier = "yGrXROHx_VazA0uovsxKfE263LMFcrSrdm4SlC-rob8"

        url = (
            "https://auth0.openai.com/authorize?client_id=pdlLIX2Y72MIl2rhLhTE9VV9bN905kBh&audience=https%3A%2F"
            "%2Fapi.openai.com%2Fv1&redirect_uri=com.openai.chat%3A%2F%2Fauth0.openai.com%2Fios%2Fcom.openai.chat"
            "%2Fcallback&scope=openid%20email%20profile%20offline_access%20model.request%20model.read"
            "%20organization.read%20offline&response_type=code&code_challenge={}"
            "&code_challenge_method=S256&prompt=login".format(code_challenge)
        )
        return self.__part_three(code_verifier, url)

    def __part_three(self, code_verifier, url: str) -> str:
        headers = {
            "User-Agent": self.user_agent,
            "Referer": "https://ios.chat.openai.com/",
        }
        resp = self.session.get(
            url, headers=headers, allow_redirects=True, **self.req_kwargs
        )

        if resp.status_code == 200:
            try:
                url_params = parse_qs(urlparse(resp.url).query)
                state = url_params["state"][0]
                return self.__part_four(code_verifier, state)
            except IndexError as exc:
                raise Exception("Rate limit hit.") from exc
        else:
            raise Exception("Error request login url.")

    def __part_four(self, code_verifier: str, state: str) -> str:
        url = "https://auth0.openai.com/u/login/identifier?state=" + state
        headers = {
            "User-Agent": self.user_agent,
            "Referer": url,
            "Origin": "https://auth0.openai.com",
        }
        data = {
            "state": state,
            "username": self.email,
            "js-available": "true",
            "webauthn-available": "true",
            "is-brave": "false",
            "webauthn-platform-available": "false",
            "action": "default",
        }
        resp = self.session.post(
            url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs
        )

        if resp.status_code == 302:
            return self.__part_five(code_verifier, state)
        else:
            raise Exception("Error check email.")

    def __part_five(self, code_verifier: str, state: str) -> str:
        url = "https://auth0.openai.com/u/login/password?state=" + state
        headers = {
            "User-Agent": self.user_agent,
            "Referer": url,
            "Origin": "https://auth0.openai.com",
        }
        data = {
            "state": state,
            "username": self.email,
            "password": self.password,
            "action": "default",
        }

        resp = self.session.post(
            url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs
        )
        if resp.status_code == 302:
            location = resp.headers["Location"]
            if not location.startswith("/authorize/resume?"):
                raise Exception("Login failed.")

            return self.__part_six(code_verifier, location, url)

        if resp.status_code == 400:
            raise Exception(f"Wrong email or password: {resp.text}")
        else:
            raise Exception("Error login.")

    def __part_six(self, code_verifier: str, location: str, ref: str) -> str:
        url = "https://auth0.openai.com" + location
        headers = {
            "User-Agent": self.user_agent,
            "Referer": ref,
        }

        resp = self.session.get(
            url, headers=headers, allow_redirects=False, **self.req_kwargs
        )
        if resp.status_code == 302:
            location = resp.headers["Location"]
            if location.startswith("/u/mfa-otp-challenge?"):
                if not self.mfa:
                    raise Exception("MFA required.")
                return self.__part_seven(code_verifier, location)

            if not location.startswith(
                "com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback?"
            ):
                raise Exception("Login callback failed.")

            return self.get_access_token(code_verifier, resp.headers["Location"])

        raise Exception("Error login.")

    def __part_seven(self, code_verifier: str, location: str) -> str:
        url = "https://auth0.openai.com" + location
        data = {
            "state": parse_qs(urlparse(url).query)["state"][0],
            "code": self.mfa,
            "action": "default",
        }
        headers = {
            "User-Agent": self.user_agent,
            "Referer": url,
            "Origin": "https://auth0.openai.com",
        }

        resp = self.session.post(
            url, headers=headers, data=data, allow_redirects=False, **self.req_kwargs
        )
        if resp.status_code == 302:
            location = resp.headers["Location"]
            if not location.startswith("/authorize/resume?"):
                raise Exception("MFA failed.")

            return self.__part_six(code_verifier, location, url)

        if resp.status_code == 400:
            raise Exception("Wrong MFA code.")
        else:
            raise Exception("Error login.")

    def get_access_token(self, code_verifier: str, callback_url: str) -> str:
        url_params = parse_qs(urlparse(callback_url).query)

        if "error" in url_params:
            error = url_params["error"][0]
            error_description = (
                url_params["error_description"][0]
                if "error_description" in url_params
                else ""
            )
            raise Exception("{}: {}".format(error, error_description))

        if "code" not in url_params:
            raise Exception("Error get code from callback url.")

        url = "https://auth0.openai.com/oauth/token"
        headers = {
            "User-Agent": self.user_agent,
        }
        data = {
            "redirect_uri": "com.openai.chat://auth0.openai.com/ios/com.openai.chat/callback",
            "grant_type": "authorization_code",
            "client_id": "pdlLIX2Y72MIl2rhLhTE9VV9bN905kBh",
            "code": url_params["code"][0],
            "code_verifier": code_verifier,
        }
        resp = self.session.post(
            url, headers=headers, json=data, allow_redirects=False, **self.req_kwargs
        )

        if resp.status_code == 200:
            json = resp.json()
            if "access_token" not in json:
                raise Exception("Get access token failed, maybe you need a proxy.")

            self.access_token = json["access_token"]
            self.expires = (
                dt.utcnow()
                + timedelta(seconds=json["expires_in"])
                - timedelta(minutes=5)
            )
            return self.access_token
        else:
            raise Exception(resp.text)


@retry(tries=3)
def get_new_accounts_1():
    response = requests.get("https://djsfenxiang.com/zhyhq.txt")
    if response.status_code == 200:
        accounts = []
        for line in response.text.splitlines():
            email, password = line.split("---")
            accounts.append(
                {
                    "Email": email,
                    "Password": password,
                }
            )
        return pd.DataFrame(accounts)
    else:
        logging.error(
            f"[get_new_accounts_1]Request failed, status code: {response.status_code}, content: {response.text}"
        )
        return None


def job():
    logging.info(f"job start")
    df = pd.read_csv(ACCOUNT_CSV_FILE)
    df_new_1 = get_new_accounts_1()
    df = pd.concat([df, df_new_1]).drop_duplicates(subset=["Email", "Password"])
    df = df.reset_index(drop=True)

    for index, row in df.iterrows():
        email = row["Email"]
        password = row["Password"]
        lastUpdateTime = row["UpdateTime"]
        if lastUpdateTime is not None and str(lastUpdateTime) != "nan":
            lastUpdateTime = datetime.strptime(lastUpdateTime, TIME_FORMAT)
            time_diff = datetime.now() - lastUpdateTime
            if time_diff.total_seconds() < 3600:
                continue
        try:
            auth = Auth0(email=email, password=password)
            access_token = auth.auth()
            if access_token != None and len(access_token) != 0:
                df.at[index, "UpdateTime"] = datetime.now().strftime(TIME_FORMAT)
                logging.info(f"account update done, email: {email}")
            else:
                df.at[index, "UpdateTime"] = None
                logging.error(
                    f"openai account error, email: {email}, error: access_token is empty"
                )
        except Exception as err:
            df.at[index, "UpdateTime"] = None
            logging.error(f"openai account error, email: {email}, error: {err}")
        time.sleep(10)  # sleep 10 seconds for api rate limit
    df = df.dropna(subset=["UpdateTime"])

    df.sort_values(by="UpdateTime", ascending=False, inplace=True)
    df = df.reset_index(drop=True)
    df.set_index(np.arange(1, len(df) + 1), inplace=True)
    df.to_csv(ACCOUNT_CSV_FILE, index=False)
    df.to_csv(
        ACCOUNT_CSV_FILE + "." + datetime.now().strftime(TIME_FORMAT), index=False
    )
    content = df.to_markdown().encode()
    auth = Auth.Token(
        TOKEN
        + "11AB7CB2I0SsHGlxDmEge9_io3PDlpl38PsjO2EmHMlRstpQ6Nb8qcgCCLNLW9vDhdU73K3W22FwjxmbMj"
    )
    g = Github(auth=auth)
    repo = g.get_user().get_repo("free-chatgpt-accounts")
    file = repo.get_contents("README.md")
    repo.update_file("README.md", "Update README.md", content, file.sha)

    logging.info(f"job end")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.INFO)

    job()

    schedule.every(12).hours.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)
