import { IORouter } from './router_registry';
import { IOHandler, ModelArtifacts, SaveResult } from './types';
export declare class BrowserHTTPRequest implements IOHandler {
    private readonly weightPathPrefix?;
    private readonly onProgress?;
    protected readonly path: string | string[];
    protected readonly requestInit: RequestInit;
    private readonly fetchFunc;
    readonly DEFAULT_METHOD = "POST";
    static readonly URL_SCHEME_REGEX: RegExp;
    constructor(path: string | string[], requestInit?: RequestInit, weightPathPrefix?: string, fetchFunc?: Function, onProgress?: Function);
    save(modelArtifacts: ModelArtifacts): Promise<SaveResult>;
    load(): Promise<ModelArtifacts>;
    private loadBinaryTopology;
    protected loadBinaryModel(): Promise<ModelArtifacts>;
    protected loadJSONModel(): Promise<ModelArtifacts>;
    private loadWeights;
    private getFetchFunc;
}
export declare function parseUrl(url: string): [string, string];
export declare function isHTTPScheme(url: string): boolean;
export declare const httpRequestRouter: IORouter;
export declare function browserHTTPRequest(path: string | string[], requestInit?: RequestInit, weightPathPrefix?: string, fetchFunc?: Function, onProgress?: Function): IOHandler;
